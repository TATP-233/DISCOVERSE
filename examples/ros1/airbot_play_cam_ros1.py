import numpy as np

import rospy
import cv_bridge
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import JointState, Image, CameraInfo

from discoverse.robots_env.airbot_play_base import AirbotPlayCfg, AirbotPlayBase

class AirbotPlayCam(AirbotPlayBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)

        self.tar_jq = np.zeros(self.nj)

        self.joint_state_puber = rospy.Publisher("/airbot_play/joint_states", JointState, queue_size=5)
        self.gripper_position_puber = rospy.Publisher("/airbot_play/gripper/position", Float64, queue_size=5)
        self.arm_joint_cmd_suber = rospy.Subscriber("/airbot_play/joint_cmd", JointState, self.arm_cmd_cb)
        self.eef_joint_cmd_suber = rospy.Subscriber("/airbot_play/end_effector/command", JointState, self.eef_cmd_cb)
        self.gripper_bool_cmd_suber = rospy.Subscriber("/airbot_play/gripper/state_cmd", Bool, self.gripper_bool_cmd_cb)
        self.gripper_float_cmd_suber = rospy.Subscriber("/airbot_play/gripper/set_position", Float64, self.gripper_float_cmd_cb)
        self.image_puber = rospy.Publisher("/camera/color/image_raw", Image, queue_size=5)
        self.camera_info_puber = rospy.Publisher("/camera/color/camera_info", CameraInfo, queue_size=5)

        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(6)] + ["endleft", "endright"]

        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

        self.js_timer = rospy.Timer(rospy.Duration(1 / 200), self.pub_joint_states)
        self.img_timer = rospy.Timer(rospy.Duration(1 / 30), self.pub_image)

    def resetState(self):
        super().resetState()
        self.tar_jq = np.zeros(self.nj)

    def updateControl(self, action):
        super().updateControl(self.tar_jq)

    def arm_cmd_cb(self, msg: JointState):
        self.tar_jq[:6] = np.array(msg.position)

    def eef_cmd_cb(self, msg: JointState):
        self.tar_jq[6] = msg.position[0]

    def gripper_bool_cmd_cb(self, msg: Bool):
        self.tar_jq[6] = 0.0 if msg.data else 1.0

    def gripper_float_cmd_cb(self, msg: Float64):
        self.tar_jq[6] = np.clip(msg.data, 0.25, 1)

    def pub_joint_states(self, event):
        self.joint_state.header.stamp = rospy.Time.now()
        qpos = self.sensor_joint_qpos.tolist()
        eef = qpos[6] * 0.025
        self.joint_state.position = qpos + [-eef]
        self.joint_state.velocity = self.sensor_joint_qvel.tolist() + [-eef]
        self.joint_state.effort = self.sensor_joint_force.tolist() + [-eef]
        self.joint_state_puber.publish(self.joint_state)
        self.gripper_position_puber.publish(Float64(data=eef))

    def pub_image(self, event):
        if self.obs is None:
            return
        image = self.obs["img"][0]
        print(image.shape, image.dtype)
        self.image_puber.publish(cv_bridge.CvBridge().cv2_to_imgmsg(image, "rgb8"))
        info = CameraInfo()
        info.header.stamp = rospy.Time.now()
        h, w, _ = image.shape
        info.width = w
        info.height = h
        self.camera_info_puber.publish(info)


if __name__ == "__main__":
    rospy.init_node("Airbot_play_mujoco_node", anonymous=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/tasks_airbot_play/stack_two_colors_of_blocks.xml"
    cfg.decimation = 4
    cfg.timestep = 0.002
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = [0]
    cfg.enable_render = True
    cfg.render_set   = {
        "fps"    : 30,
        "width"  : 640,
        "height" : 480
    }
    exec_node = AirbotPlayCam(cfg)
    while not rospy.is_shutdown() and exec_node.running:
        exec_node.step()
