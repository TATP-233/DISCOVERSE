"""
opencv-python
mediapipe
pykinect_azure
numpy
scipy

kinect:
https://blog.csdn.net/weixin_42283539/article/details/130621234
"""

import cv2
import mediapipe as mp
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, k4a_float2_t

import numpy as np
from google.protobuf.json_format import MessageToDict


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

    def process_frame(self, color_img, depth_img):
        h, w = color_img.shape[0], color_img.shape[1]

        img_RGB = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_RGB)
        joints_all = []
        joints_ref_all = []
        label_list = []

        if results.multi_hand_landmarks:
            for hand_idx in range(len(results.multi_hand_landmarks)):
                joints = []
                joints_ref = []

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

                joints_all.append(joints)
                joints_ref_all.append(joints_ref)
                label_list.append(label)

        return color_img, joints_all, joints_ref_all, label_list


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    k4a = k4a_driver()

    cv2.namedWindow("Space isolation teleoperation")
    while True:

        capture = k4a.device.update()
        ret_color, color_img = capture.get_color_image()
        ret_depth, depth_img = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue

        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)

        color_img, joints_all, joints_ref_all, label_list = k4a.process_frame(
            color_img, depth_img
        )

        cv2.imshow("Space isolation teleoperation", color_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    import matplotlib.pyplot as plt

    def plot_joints(joints, label):
        js = np.array(joints)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(js[:, 0], js[:, 1], js[:, 2])
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    plot_joints(joints_all[0], label_list[0])
