from discoverse.examples.hardware_in_loop.airbot_arm import AirbotArm, cfg

import rospy
import threading
from sensor_msgs.msg import JointState


class AirbotPlayShortROS1(AirbotArm):
    def __init__(self, config):
        super().__init__(config)
        rospy.init_node("Airbot_play_short_node", anonymous=True)

        self.joint_state_puber = rospy.Publisher(
            f"/airbot_{config.arm_type}/joint_states", JointState, queue_size=5
        )
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(self.nj)]
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

    def thread_pubros2topic(self, freq=30):
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown() and self.running:
            self.joint_state.header.stamp = rospy.Time.now()
            self.joint_state.position = self.sensor_joint_qpos.tolist()
            self.joint_state.velocity = self.sensor_joint_qvel.tolist()
            self.joint_state.effort = self.sensor_joint_force.tolist()
            self.joint_state_puber.publish(self.joint_state)
            rate.sleep()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run arm with specified parameters. \ne.g. python3 airbot_play_short.py --arm_type play_short --eef_type none"
    )
    parser.add_argument(
        "--arm_type",
        type=str,
        choices=["play_long", "play_short", "lite", "pro", "replay"],
        help="Name of the arm",
        default="play_short",
    )
    parser.add_argument(
        "--eef_type",
        type=str,
        choices=["G2", "E2B", "PE2", "none"],
        help="Name of the eef",
        default="none",
    )
    parser.add_argument(
        "--discoverse_viewer", action="store_true", help="Use discoverse viewer"
    )
    args = parser.parse_args()

    cfg.arm_type = args.arm_type
    cfg.eef_type = args.eef_type
    if not args.discoverse_viewer:
        import mujoco.viewer

        cfg.enable_render = False

    exec_node = AirbotPlayShortROS1(cfg)

    obs = exec_node.reset()

    def func_while_running():
        exec_node.action[: exec_node.nj] = 0.15
        obs, pri_obs, rew, ter, info = exec_node.step()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    cfg.sync = False
    rate = rospy.Rate(int(1.0 / exec_node.delta_t))
    if args.discoverse_viewer:
        while not rospy.is_shutdown() and exec_node.running:
            func_while_running()
            rate.sleep()
    else:
        with mujoco.viewer.launch_passive(
            exec_node.mj_model,
            exec_node.mj_data,
            key_callback=exec_node.windowKeyPressCallback,
        ) as viewer:
            while viewer.is_running():
                func_while_running()
                viewer.sync()
                rate.sleep()

    pubtopic_thread.join()
