import sys
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from discoverse.envs.mmk2_base import MMK2Cfg, MMK2Base


class RobotArmSync(MMK2Base):
    def __init__(self, config: MMK2Cfg, robot_name: str):
        super().__init__(config)
        self.mj_base = self.mj_data.qpos[:7]
        self.mj_wheel_joint = self.mj_data.qpos[7:9]
        self.mj_slide_joint = self.mj_data.qpos[9:10]
        self.mj_head_joint = self.mj_data.qpos[10:12]
        self.mj_left_arm_joint = self.mj_data.qpos[12:18]
        self.mj_left_gripper_joint = self.mj_data.qpos[18:20]
        self.mj_right_arm_joint = self.mj_data.qpos[20:26]
        self.mj_right_gripper_joint = self.mj_data.qpos[26:28]

        self.left_joint_suber = rospy.Subscriber(
            f"/{robot_name}/arms/left/joint_states",
            JointState,
            self.left_joint_callback,
        )
        self.right_joint_suber = rospy.Subscriber(
            f"/{robot_name}/arms/left/joint_states",
            JointState,
            self.right_joint_callback,
        )

    def left_joint_callback(self, msg: JointState):
        if len(msg.position) == 7:
            self.mj_left_arm_joint[:] = msg.position[:6]
            self.mj_left_gripper_joint[0] = 0.04 * msg.position[6]
            self.mj_left_gripper_joint[1] = -0.04 * msg.position[6]
        else:
            rospy.logwarn(
                f"Invalid <LEFT> arm joint state message, please check. msg.position = {np.array(msg.position)}"
            )

    def right_joint_callback(self, msg: JointState):
        if len(msg.position) == 7:
            self.mj_right_arm_joint[:] = msg.position[:6]
            self.mj_right_gripper_joint[0] = 0.04 * msg.position[6]
            self.mj_right_gripper_joint[1] = -0.04 * msg.position[6]
        else:
            rospy.logwarn(
                f"Invalid <RIGHT> arm joint state message, please check. msg.position = {np.array(msg.position)}"
            )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        robot_name = sys.argv[1]
    else:
        print("Usage: python teachbag_sync.py <robot_name>")
        sys.exit(0)

    rospy.init_node(f"{robot_name}_sync_mujoco_node", anonymous=True)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.render_set = {
        "fps": 30,
        "width": 1280,
        "height": 720,
    }

    exec_node = RobotArmSync(cfg, robot_name)
    exec_node.reset()

    while exec_node.running:
        exec_node.view()
