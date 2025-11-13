import os
import numpy as np

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.robots import AirbotPlayIK
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg

if __name__ == "__main__":
    import rospy
    from sensor_msgs.msg import JointState

    rospy.init_node("airbot_play", anonymous=True)
    joint_puber = rospy.Publisher("/airbot_play/joint_state", JointState, queue_size=10)
    joint_msg = JointState()

    cfg = AirbotPlayCfg()

    cfg.timestep = 1e-3
    cfg.decimation = 4
    cfg.render_set = {
        "fps"    : 60,
        "width"  : 1920,
        "height" : 1080,
    }
    cfg.init_joint_pose = {
        "joint1"  :  0.0,
        "joint2"  : -0.476,
        "joint3"  :  0.399,
        "joint4"  : -1.5708,
        "joint5"  : -0.077,
        "joint6"  :  1.570,
        "gripper" :  0.5
    }

    cfg.mjcf_file_path = "mjcf/airbot_play_floor.xml"
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = "scene/qz11/qz_table.ply"

    exec_node = AirbotPlayBase(cfg)

    obs = exec_node.reset()
    joint_msg.name = [f"joint_{i}" for i in range(exec_node.nj)]
    joint_msg.effort = np.zeros_like(obs["jq"]).tolist()

    action = exec_node.init_joint_pose[:exec_node.nj]

    arm_ik = AirbotPlayIK()

    trans = np.array([ 0.35, -0., 0.3])
    rot   = np.array([
        [ 1., -0., -0.],
        [ 0.,  1., -0.],
        [ 0.,  0.,  1.],
    ])

    cnt = 0
    while exec_node.running:
        if cnt * exec_node.delta_t < 1:
            trans[1] += 0.1 * exec_node.delta_t / 1
        elif cnt * exec_node.delta_t < 2.:
            trans[1] -= 0.2 * exec_node.delta_t / 1
        elif cnt * exec_node.delta_t < 3:
            trans[1] += 0.1 * exec_node.delta_t / 1.
            trans[2] += 0.15 * exec_node.delta_t / 1.
        elif cnt * exec_node.delta_t < 4.:
            trans[2] -= 0.3 * exec_node.delta_t / 1.
        elif cnt * exec_node.delta_t < 5:
            trans[2] += 0.15 * exec_node.delta_t / 1.
            trans[0] += 0.15 * exec_node.delta_t / 1.
        elif cnt * exec_node.delta_t < 6.:
            trans[0] -= 0.15 * exec_node.delta_t / 1.
        else:
            cnt = 0
        cnt += 1

        action[:6] = arm_ik.properIK(trans, rot, action[:6])

        obs, pri_obs, rew, ter, info = exec_node.step(action)
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.position = obs["jq"]
        joint_msg.velocity = obs["jv"]

        joint_puber.publish(joint_msg)
