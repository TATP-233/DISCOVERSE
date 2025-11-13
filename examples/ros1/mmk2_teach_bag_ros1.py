import sys
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from discoverse.robots_env.mmk2_base import MMK2Cfg, MMK2Base

class RobotArmSync(MMK2Base):
    target_control = np.zeros(19)
    def __init__(self, config: MMK2Cfg, robot_name:str):
        super().__init__(config)

        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        self.lft_arm_suber = rospy.Subscriber(f"/{robot_name}/arms/left/joint_states", JointState, self.lft_arm_callback)
        self.rgt_arm_suber = rospy.Subscriber(f"/{robot_name}/arms/left/joint_states", JointState, self.rgt_arm_callback)

    def lft_arm_callback(self, msg:JointState):
        if len(msg.position) == 7:
            self.tctr_left_arm[:] = msg.position[:6]
            self.tctr_lft_gripper[0] = msg.position[6]
        else:
            rospy.logwarn(f"Invalid <LEFT> arm joint state message, please check. msg.position = {np.array(msg.position)}")

    def rgt_arm_callback(self, msg:JointState):
        if len(msg.position) == 7:
            self.tctr_right_arm[:] = msg.position[:6]
            self.tctr_rgt_gripper[0] = msg.position[6]
        else:
            rospy.logwarn(f"Invalid <RIGHT> arm joint state message, please check. msg.position = {np.array(msg.position)}")

if __name__ == "__main__":

    if len(sys.argv) > 1:
        robot_name = sys.argv[1]
    else:
        print("Usage: python teachbag_sync.py <robot_name>")
        sys.exit(0)
    
    rospy.init_node(f'{robot_name}_sync_mujoco_node', anonymous=True)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.render_set     = {
        "fps"    :   30,
        "width"  : 1280,
        "height" :  720,
    }

    exec_node = RobotArmSync(cfg, robot_name)
    exec_node.reset()

    while exec_node.running:
        exec_node.view()
