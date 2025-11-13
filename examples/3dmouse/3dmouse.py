import cv2
import numpy as np

from scipy.spatial.transform import Rotation

from discoverse.robots import AirbotPlayIK
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg
from discoverse.utils import step_func

# refer the configuration at https://pypi.org/project/pyspacemouse/
import pyspacemouse 

if __name__ == "__main__":
    cfg = AirbotPlayCfg()

    cfg.timestep = 1e-3
    cfg.obs_rgb_cam_id = [0]
    cfg.decimation = 4
    cfg.render_set = {
        "fps"    : 60,
        "width"  : 1280,
        "height" : 720,
    }

    cfg.mjcf_file_path = "mjcf/manipulator/robot_airbot_play.xml"
    cfg.use_gaussian_renderer = False
    exec_node = AirbotPlayBase(cfg)

    arm_ik = AirbotPlayIK()
    target_position = np.array([0.280, -0., 0.220])
    target_euler = np.array([0., 0., 0.])
    target_action = np.zeros(7)

    obs = exec_node.reset()

    action = exec_node.init_joint_pose[:exec_node.nj]
    cv2.namedWindow("img0", cv2.WINDOW_GUI_NORMAL)

    success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)

    ratio_postion = 0.001
    ratio_euler = 0.003
    button_flag = False

    if success:
        while exec_node.running:
            # target_action = action
            state = pyspacemouse.read()
            # convert spacemouse state to pose
            delta_position = np.array([state.y, -state.x, state.z])
            delta_euler = np.array([-state.roll, -state.pitch, state.yaw])
            for i in range(3):
                if abs(delta_euler[i]) < 0.35:
                    delta_euler[i] = 0
            for i in range(3):
                if abs(delta_position[i]) < 0.25:
                    delta_position[i] = 0
            if state.buttons[0]:
                if button_flag == False:
                    button_flag = True
                    target_action[6] = 1 - target_action[6]
            else:
                button_flag = False
            # convert pose to action
            target_position += delta_position * ratio_postion
            target_euler += delta_euler * ratio_euler
            try:
                target_action[:6] = arm_ik.properIK(target_position, Rotation.from_euler("xyz", target_euler).as_matrix().T, action[:6])
            except:
                # print(target_position, Rotation.from_euler("xyz", target_euler).as_matrix().T)
                print("inverse kinematics failed")
                target_position -= delta_position * ratio_postion
                target_euler -= delta_euler * ratio_euler
            for i in range(exec_node.nj-1):
                action[i] = step_func(action[i], target_action[i], 20. * exec_node.delta_t)
            action[exec_node.nj-1] = target_action[exec_node.nj-1]
            obs, _, _, _, _ = exec_node.step(action)

            cv2.imshow("img0", cv2.cvtColor(obs["img"][0], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    cv2.destroyAllWindows()
