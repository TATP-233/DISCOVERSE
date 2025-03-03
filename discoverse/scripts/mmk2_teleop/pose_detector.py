import cv2
import mediapipe as mp
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, k4a_float2_t

import tf
import rospy
import tf2_ros
from geometry_msgs.msg import QuaternionStamped, TransformStamped

import numpy as np
from scipy.spatial.transform import Rotation
from google.protobuf.json_format import MessageToDict
from std_msgs.msg import Float32

from headpose import HeadPose
from utils.inference import draw_axis

from face_landmark import HeadLandmarks, draw_landmarks_on_image


class k4a_driver:
    def __init__(self, model_path="/usr/lib/x86_64-linux-gnu/libk4a.so"):
        pykinect.initialize_libraries(model_path)
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        self.depth_scale = [0.25, 1.0]

        self.device = pykinect.start_device(config=self.device_config)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.hand_landmark_colors = [
            [128, 166, 236],
            [242, 199, 122],
            [161, 85, 253],
            [110, 113, 227],
            [240, 47, 140],
            [60, 155, 223],
        ]

        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        Rmat_world_align = np.array(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
        ).T

        self.camera_frame = "kinect_camera"
        tfs = TransformStamped()
        tfs.header.frame_id = "world"
        tfs.header.stamp = rospy.Time.now()
        tfs.header.seq = 0
        tfs.child_frame_id = self.camera_frame

        tfs.transform.translation.x = 0
        tfs.transform.translation.y = 0
        tfs.transform.translation.z = 0.6

        quat = Rotation.from_matrix(Rmat_world_align).as_quat()
        tfs.transform.rotation.x = quat[0]
        tfs.transform.rotation.y = quat[1]
        tfs.transform.rotation.z = quat[2]
        tfs.transform.rotation.w = quat[3]
        self.static_broadcaster.sendTransform(tfs)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.pub_left = rospy.Publisher("Left_finger", Float32, queue_size=2)
        self.pub_right = rospy.Publisher("Right_finger", Float32, queue_size=2)
        self.pub_head_pose = rospy.Publisher(
            "Head_orientation", QuaternionStamped, queue_size=2
        )

        base_joint_names = [
            "wrist",
            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
            "thumb_tip",
            "index_finger_mcp",
            "index_finger_pip",
            "index_finger_dip",
            "index_finger_tip",
            "middle_finger_mcp",
            "middle_finger_pip",
            "middle_finger_dip",
            "middle_finger_tip",
            "ring_finger_mcp",
            "ring_finger_pip",
            "ring_finger_dip",
            "ring_finger_tip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip",
        ]
        self.specified_name_left = [str("Left_") + i for i in base_joint_names]
        self.specified_name_right = [str("Right_") + i for i in base_joint_names]

    def process_frame(self, color_img, depth_img):
        h, w = color_img.shape[0], color_img.shape[1]

        img_RGB = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_RGB)
        joints_ref_all = []
        label_list = []
        orientation_list = []
        finger_distance_list = []

        if results.multi_hand_landmarks:
            for hand_idx in range(len(results.multi_hand_landmarks)):
                joints = []
                joints_ref = []
                finger_distance = []

                label = MessageToDict(results.multi_handedness[hand_idx])[
                    "classification"
                ][0]["label"]
                hand_21 = results.multi_hand_landmarks[hand_idx]
                self.mpDraw.draw_landmarks(
                    color_img, hand_21, self.mp_hands.HAND_CONNECTIONS
                )
                cz0 = hand_21.landmark[0].z
                for i in range(21):
                    cx = int(hand_21.landmark[i].x * w)
                    cy = int(hand_21.landmark[i].y * h)
                    if 0 < cx < w and 0 < cy < h:
                        depth = depth_img[cy, cx]
                        pixels = k4a_float2_t((cx, cy))
                        pos3d_color = self.device.calibration.convert_2d_to_3d(
                            pixels,
                            depth,
                            K4A_CALIBRATION_TYPE_COLOR,
                            K4A_CALIBRATION_TYPE_COLOR,
                        )

                        joints.append(
                            [
                                pos3d_color.xyz.x * 1e-3,
                                pos3d_color.xyz.y * 1e-3,
                                pos3d_color.xyz.z * 1e-3,
                            ]
                        )
                        joints_ref.append(
                            [
                                results.multi_hand_world_landmarks[hand_idx]
                                .landmark[i]
                                .x,
                                results.multi_hand_world_landmarks[hand_idx]
                                .landmark[i]
                                .y,
                                results.multi_hand_world_landmarks[hand_idx]
                                .landmark[i]
                                .z,
                            ]
                        )
                        cz = hand_21.landmark[i].z
                        depth_z = cz0 - cz

                        if i in {4, 8}:
                            finger_distance.append(np.array([cx, cy, cz]))
                        radius = max(int(9 * (1 + depth_z * 5)), 0)

                        if i == 0:  # 手腕
                            color_img = cv2.circle(
                                color_img,
                                (cx, cy),
                                radius,
                                self.hand_landmark_colors[0],
                                -1,
                            )
                        elif i == 8:  # 食指指尖
                            color_img = cv2.circle(
                                color_img,
                                (cx, cy),
                                radius,
                                self.hand_landmark_colors[1],
                                -1,
                            )
                        elif i in {1, 5, 9, 13, 17}:  # 指根
                            color_img = cv2.circle(
                                color_img,
                                (cx, cy),
                                radius,
                                self.hand_landmark_colors[2],
                                -1,
                            )
                        elif i in {2, 6, 10, 14, 18}:  # 第一指节
                            color_img = cv2.circle(
                                color_img,
                                (cx, cy),
                                radius,
                                self.hand_landmark_colors[3],
                                -1,
                            )
                        elif i in {3, 7, 11, 15, 19}:  # 第二指节
                            color_img = cv2.circle(
                                color_img,
                                (cx, cy),
                                radius,
                                self.hand_landmark_colors[4],
                                -1,
                            )
                        elif i in {4, 12, 16, 20}:  # 指尖（除食指指尖）
                            color_img = cv2.circle(
                                color_img,
                                (cx, cy),
                                radius,
                                self.hand_landmark_colors[5],
                                -1,
                            )
                    else:
                        break

                if len(joints) == 21 and joints[0][2] > 0:
                    distance = np.linalg.norm(finger_distance[0] - finger_distance[1])
                    if distance < 200:
                        for i in range(1, 21):
                            if i in {5, 17}:
                                joints_ref[i][0] = joints[i][0]
                                joints_ref[i][1] = joints[i][1]
                                joints_ref[i][2] = joints[i][2]
                            else:
                                joints_ref[i][0] += joints[0][0] - joints_ref[0][0]
                                joints_ref[i][1] += joints[0][1] - joints_ref[0][1]
                                joints_ref[i][2] += joints[0][2] - joints_ref[0][2]
                        joints_ref[0][0] = joints[0][0]
                        joints_ref[0][1] = joints[0][1]
                        joints_ref[0][2] = joints[0][2]

                        try:
                            tri = np.array(joints_ref)[[0, 5, 17]].copy()
                            p_z = np.mean(tri[1:3], axis=0)
                            arr_oz = p_z - tri[0]
                            if label == "Left":
                                arr_oy = tri[1] - tri[2]
                            else:
                                arr_oy = tri[2] - tri[1]

                            arr_ox = np.cross(arr_oy, arr_oz)
                            arr_ox /= np.linalg.norm(arr_ox)
                            arr_oz /= np.linalg.norm(arr_oz)

                            arr_oy = np.cross(arr_oz, arr_ox)

                            rmat = np.zeros((3, 3))
                            rmat[:, 0] = arr_ox
                            rmat[:, 1] = arr_oy
                            rmat[:, 2] = arr_oz

                            cx = int(hand_21.landmark[0].x * w)
                            cy = int(hand_21.landmark[0].y * h)
                            size = 100
                            cv2.line(
                                color_img,
                                (int(cx), int(cy)),
                                (
                                    int(size * arr_ox[0] + cx),
                                    int(size * arr_ox[1] + cy),
                                ),
                                (0, 0, 255),
                                2,
                            )
                            cv2.line(
                                color_img,
                                (int(cx), int(cy)),
                                (
                                    int(size * arr_oy[0] + cx),
                                    int(size * arr_oy[1] + cy),
                                ),
                                (0, 255, 0),
                                2,
                            )
                            cv2.line(
                                color_img,
                                (int(cx), int(cy)),
                                (
                                    int(size * arr_oz[0] + cx),
                                    int(size * arr_oz[1] + cy),
                                ),
                                (255, 0, 0),
                                2,
                            )
                        except:
                            continue

                        n_arr_x = rmat[:, 0]
                        n_arr_z = rmat[:, 2]
                        n_arr_x[2] *= -1.0
                        n_arr_z[2] *= -1.0
                        n_arr_y = np.cross(n_arr_z, n_arr_x)
                        nrmat = np.zeros((3, 3))
                        nrmat[:, 0] = n_arr_x
                        nrmat[:, 1] = n_arr_y
                        nrmat[:, 2] = n_arr_z

                        quat = Rotation.from_matrix(nrmat).as_quat()

                        # euler_ypr = Rotation.from_quat(quat).as_euler("zyx")
                        # normal_str = f'{euler_ypr[0]:6.2f}, {euler_ypr[1]:6.2f}, {euler_ypr[2]:6.2f}'
                        # cv2.putText(color_img, f'{label[0]}: {normal_str}', (10, 40*hand_idx+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        joints_ref_all.append(joints_ref)
                        label_list.append(label)
                        orientation_list.append(quat)
                        finger_distance_list.append(distance)

        return (
            color_img,
            joints_ref_all,
            label_list,
            orientation_list,
            finger_distance_list,
        )

    def publish(self, hnd, joint, quat, distance):
        specified_name = (
            self.specified_name_left if hnd == "Left" else self.specified_name_right
        )

        if len(joint) == len(specified_name):
            for i in range(len(specified_name)):
                self.tf_broadcaster.sendTransform(
                    [joint[i][0], joint[i][1], self.depth_scale[1] - (joint[i][2])],
                    quat.tolist(),
                    rospy.Time.now(),
                    specified_name[i],
                    self.camera_frame,
                )
            if hnd == "Left":
                self.pub_left.publish(distance)
            elif hnd == "Right":
                self.pub_right.publish(distance)


if __name__ == "__main__":
    import time

    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    rospy.init_node("socket_server", anonymous=True)

    k4a = k4a_driver()
    hp = HeadPose()
    hl = HeadLandmarks()

    cv2.namedWindow("Space isolation teleoperation")
    while not rospy.is_shutdown():
        # start_time = time.time()

        capture = k4a.device.update()
        ret_color, color_img = capture.get_color_image()
        ret_depth, depth_img = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue

        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)

        color_img, poses = hp.process_frame(color_img, if_draw_axis=False)

        detection_result = hl.proc(color_img)
        color_img = draw_landmarks_on_image(color_img, detection_result)
        (
            color_img,
            joints_all,
            label_list,
            orientation_list,
            distance_list,
        ) = k4a.process_frame(color_img, depth_img)

        if not poses is None:
            head_angles = poses[0][0]
            head_quat = Rotation.from_euler("zyx", head_angles, degrees=True).as_quat()
            quat_msg = QuaternionStamped()
            quat_msg.header.stamp = rospy.Time.now()
            quat_msg.quaternion.x = head_quat[0]
            quat_msg.quaternion.y = head_quat[1]
            quat_msg.quaternion.z = head_quat[2]
            quat_msg.quaternion.w = head_quat[3]
            k4a.pub_head_pose.publish(quat_msg)
            for angles, translation, lmks in poses:
                color_img = draw_axis(
                    color_img,
                    angles[0],
                    angles[1],
                    angles[2],
                    translation[0],
                    translation[1],
                    size=50,
                    pts68=lmks,
                )

        # end_time = time.time()
        # scaler = 1
        # FPS = 1./(end_time - start_time)
        # img_axis_plot = cv2.putText(color_img, 'FPS ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (0, 0, 255), 2 * scaler)

        # simg = cv2.resize(color_img, (1280, 720))
        cv2.imshow("Space isolation teleoperation", color_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        try:
            if len(joints_all) == 2:
                if (
                    np.linalg.norm(
                        np.array(joints_all[0])[0] - np.array(joints_all[1])[0]
                    )
                    > 0.06
                ):
                    k4a.publish(
                        label_list[0],
                        joints_all[0],
                        orientation_list[0],
                        distance_list[0],
                    )
                    k4a.publish(
                        label_list[1],
                        joints_all[1],
                        orientation_list[1],
                        distance_list[1],
                    )
            elif len(joints_all) == 1:
                k4a.publish(
                    label_list[0], joints_all[0], orientation_list[0], distance_list[0]
                )
        except Exception as e:
            print(
                "-------------------------------------------------------------------------------------"
            )
            print(e)
            print(
                "joints_all       = \\\n",
                np.array2string(np.array(joints_all), separator=", "),
            )
            print(
                "label_list       =\n",
                np.array2string(np.array(label_list), separator=", "),
            )
            print("orientation_list =\n", orientation_list)
            print(
                "distance_list    =\n",
                np.array2string(np.array(distance_list), separator=", "),
            )
            continue
