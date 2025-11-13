import os
import cv2
import glfw
import time
import numpy as np

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
from skyrover_on_car_base import SkyRoverOnCarBase
from cooperative_control import CooperativeAerialGroundControl
import json

class SkyRoverOnCarCfg(BaseConfig):
    # mjcf_file_path = "mjcf/skyrover_on_rm2_car.xml"
    # mjcf_file_path = "mjcf/cooperative_aerial_ground_sim.xml"
    mjcf_file_path = "mjcf/cooperative_multi_robot_sim.xml"
    timestep       = 1/240 #0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 60, #30,
        "width"  : 1920,
        "height" : 1080 
    }
    obs_rgb_cam_id   = [-1, 0, 1, 2, 3, 4]
    obs_depth_cam_id = None
    rb_link_list   = [
        ########################### skyrover ###########################
        "skyrover",
        "skyrover_base_link",
        "skyrover_folder1_link",
        "skyrover_folder2_link",
        "skyrover_wheel1_link",
        "skyrover_wheel2_link",
        "skyrover_wheel3_link",
        "skyrover_wheel4_link",
        "skyrover_rotor1_link",
        "skyrover_rotor2_link",
        "skyrover_rotor3_link",
        "skyrover_rotor4_link",
        ############################## car ##############################
        # "rm2" , "wheel1", "wheel2", "wheel3", "wheel4" , 
        # "steering1", "steering2", "steering3", "steering4",

        "rm2" , "rm2_wheel1", "rm2_wheel2", "rm2_wheel3", "rm2_wheel4" , 
        "rm2_steering1", "rm2_steering2", "rm2_steering3", "rm2_steering4",

        # BG

        "rm2_1" , 
        "rm2_1_wheel1", 
        "rm2_1_wheel2", 
        "rm2_1_wheel3", 
        "rm2_1_wheel4" , 
        "rm2_1_steering1", 
        "rm2_1_steering2", 
        "rm2_1_steering3", 
        "rm2_1_steering4",

        "rm2_2" , 
        "rm2_2_wheel1", 
        "rm2_2_wheel2", 
        "rm2_2_wheel3", 
        "rm2_2_wheel4" , 
        "rm2_2_steering1", 
        "rm2_2_steering2", 
        "rm2_2_steering3", 
        "rm2_2_steering4",
    ]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict  = {
        "skyrover"              :   "skyrover/stretch_link.ply",
        "skyrover_base_link"    :   "skyrover/skyrover_base.ply",
        "skyrover_folder1_link" :   "skyrover/folder1_link.ply",
        "skyrover_folder2_link" :   "skyrover/folder2_link.ply",
        "skyrover_wheel2_link"  :   "skyrover/wheel1_2.ply",
        "skyrover_wheel1_link"  :   "skyrover/wheel1_2.ply",
        "skyrover_wheel3_link"  :   "skyrover/wheel3_4.ply",
        "skyrover_wheel4_link"  :   "skyrover/wheel3_4.ply",
        "skyrover_rotor2_link"  :   "skyrover/rotor2_4.ply",
        "skyrover_rotor1_link"  :   "skyrover/rotor1_3.ply",
        "skyrover_rotor3_link"  :   "skyrover/rotor1_3.ply",
        "skyrover_rotor4_link"  :   "skyrover/rotor2_4.ply",    

        # "rm2"                   :   "rm2_car/sundial/rm2.ply",
        # "wheel1"                :   "rm2_car/sundial/rgt_wheel.ply",
        # "wheel2"                :   "rm2_car/sundial/lft_wheel.ply",
        # "wheel3"                :   "rm2_car/sundial/lft_wheel.ply",
        # "wheel4"                :   "rm2_car/sundial/rgt_wheel.ply",
        # "steering1"             :   "rm2_car/sundial/steering_left_front.ply",
        # "steering2"             :   "rm2_car/sundial/steering_right_front.ply",
        # "steering3"             :   "rm2_car/sundial/steering_left_back.ply",
        # "steering4"             :   "rm2_car/sundial/steering_right_back.ply",

        "rm2"                       :   "rm2_car/sundial/rm2.ply",
        "rm2_wheel1"                :   "rm2_car/sundial/rgt_wheel.ply",
        "rm2_wheel2"                :   "rm2_car/sundial/lft_wheel.ply",
        "rm2_wheel3"                :   "rm2_car/sundial/lft_wheel.ply",
        "rm2_wheel4"                :   "rm2_car/sundial/rgt_wheel.ply",
        "rm2_steering1"             :   "rm2_car/sundial/steering_left_front.ply",
        "rm2_steering2"             :   "rm2_car/sundial/steering_right_front.ply",
        "rm2_steering3"             :   "rm2_car/sundial/steering_left_back.ply",
        "rm2_steering4"             :   "rm2_car/sundial/steering_right_back.ply",

        # BG

        "rm2_1"                       :   "rm2_car/sundial/rm2.ply",
        "rm2_1_wheel1"                :   "rm2_car/sundial/rgt_wheel.ply",
        "rm2_1_wheel2"                :   "rm2_car/sundial/lft_wheel.ply",
        "rm2_1_wheel3"                :   "rm2_car/sundial/lft_wheel.ply",
        "rm2_1_wheel4"                :   "rm2_car/sundial/rgt_wheel.ply",
        "rm2_1_steering1"             :   "rm2_car/sundial/steering_left_front.ply",
        "rm2_1_steering2"             :   "rm2_car/sundial/steering_right_front.ply",
        "rm2_1_steering3"             :   "rm2_car/sundial/steering_left_back.ply",
        "rm2_1_steering4"             :   "rm2_car/sundial/steering_right_back.ply",

        "rm2_2"                       :   "rm2_car/sundial/rm2.ply",
        "rm2_2_wheel1"                :   "rm2_car/sundial/rgt_wheel.ply",
        "rm2_2_wheel2"                :   "rm2_car/sundial/lft_wheel.ply",
        "rm2_2_wheel3"                :   "rm2_car/sundial/lft_wheel.ply",
        "rm2_2_wheel4"                :   "rm2_car/sundial/rgt_wheel.ply",
        "rm2_2_steering1"             :   "rm2_car/sundial/steering_left_front.ply",
        "rm2_2_steering2"             :   "rm2_car/sundial/steering_right_front.ply",
        "rm2_2_steering3"             :   "rm2_car/sundial/steering_left_back.ply",
        "rm2_2_steering4"             :   "rm2_car/sundial/steering_right_back.ply",
    }

class SkyRoverOnCarBase(SimulatorBase):
    def updateControl(self, action):
        self.mj_data.ctrl[:] = action[:]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq"  : self.mj_data.qpos.tolist(),
            "jv"  : self.mj_data.qvel.tolist(),
            "img" : self.img_rgb_obs_s,
            "dep" : self.img_depth_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None
    
    def on_key(self, window, key, scancode, action, mods):
        super().on_key(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            if key == glfw.KEY_S:
                for camid, img in self.obs["img"].items():
                    print(self.getCameraPose(camid))
                    cv2.imwrite(os.path.join(DISCOVERSE_ROOT_DIR, f"img_{camid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                for camid, img in self.obs["dep"].items():
                    np.save(os.path.join(DISCOVERSE_ROOT_DIR, f"dep_{camid}.npy"), img)
                    print(img.max(), img.min())

if __name__ == "__main__":

    cfg = SkyRoverOnCarCfg()
    cfg.use_gaussian_renderer = True

    cfg.gs_model_dict["background"] = "scene/riverside1/point_cloud.ply"
    cfg.gs_model_dict["background_env"] = "scene/riverside1/environment.ply"

    cfg.obs_rgb_cam_id = [-1]
    cfg.obs_depth_cam_id = [-1]

    exec_node = SkyRoverOnCarBase(cfg)
    obs = exec_node.reset()

    cc = CooperativeAerialGroundControl(exec_node)
    mj_control = np.zeros(15 + 8)
    simulation_duration = 500
    start_time = time.time()

    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(exec_node.mj_model,
                                          exec_node.mj_data)
    
    time_data = []
    qpos_data = []
    steps = 0

    img_lst_third_view = []
    img_lst_first_person_front = []
    img_lst_first_person_down = []

    sine_motion_start_time = time.time()

    while exec_node.running:
        """ Simulation timer """
        sim_time = time.time() - start_time
        if sim_time >= simulation_duration:
            print("Simulation duration reached. Exiting...")
            break

        """ Step timer """
        step_start = time.time()

        """ Mission execution """
        cc.target_pos = [0., 0., 0.] # Dummy (for logging)
        if cc.do_mission(mission_phase=cc.mission_phase):
            break

        """ Fake quadrotors in background """
        # Drone 1
        exec_node.mj_data.qpos[63] = 10.0
        exec_node.mj_data.qpos[64] = 3*np.sin(0.1*(time.time()-sine_motion_start_time))
        exec_node.mj_data.qpos[65] = 8.0
        exec_node.mj_data.qpos[66] = 1.0
        exec_node.mj_data.qpos[67] = 0.0
        exec_node.mj_data.qpos[68] = 0.0
        exec_node.mj_data.qpos[69] = 0.0

        # Drone 2 
        exec_node.mj_data.qpos[81] = 10.0
        exec_node.mj_data.qpos[82] = -2*np.sin(0.1*(time.time()-sine_motion_start_time))
        exec_node.mj_data.qpos[83] = 6.0
        exec_node.mj_data.qpos[84] = 1.0
        exec_node.mj_data.qpos[85] = 0.0
        exec_node.mj_data.qpos[86] = 0.0
        exec_node.mj_data.qpos[87] = 0.0

        """ Data logging """
        skyrover_pos, skyrover_vel, skyrover_quat, skyrover_pqr, skyrover_eul = cc.multi_robot.get_skyrover_state(verbose=False)
        # rm2_pos, rm2_vel, _, rm2_pqr, rm2_eul = cc.multi_robot.get_rm2_state(verbose=False)
        cc.logger.log_quadrotor_state(skyrover_pos, skyrover_vel, skyrover_quat, skyrover_pqr, skyrover_eul, 
                            cc.target_pos, [0. ,0. ,0.], [0. ,0. ,0.], [0. ,0. ,0.], [0. ,0. ,0.])
        current_time = steps * exec_node.mj_model.opt.timestep
        time_data.append(current_time)
        qpos_data.append(exec_node.mj_data.qpos.tolist())

        # obs, pri_obs, rew, ter, info = exec_node.step(mj_control)
        obs, _, _, _, _ = exec_node.step(exec_node.mj_data.ctrl)

        # mujoco.mj_step(exec_node.mj_model, exec_node.mj_data)
        # viewer.sync()

        steps += 1

        img_lst_third_view.append(exec_node.obs["img"][-1])
        img_lst_first_person_front.append(exec_node.obs["img"][1])
        img_lst_first_person_down.append(exec_node.obs["img"][2])

    """ Save results """
    cc.logger.plot_states()
    cc.logger.plot_3d_position()

    # Prepare the JSON data
    json_data = {
        "time": time_data,
        "qpos": qpos_data
    }

    # Save to JSON
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("discoverse/examples/skyrover_on_rm2car/log/%Y-%m-%d %H:%M:%S")
    filename = f"{timestamp}.json"
    with open(filename, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    import mediapy
    mediapy.write_video(f"discoverse/examples/skyrover_on_rm2car/log/%Y-%m-%d %H:%M:%S_third_view.mp4", img_lst_third_view, fps=30)
    mediapy.write_video(f"discoverse/examples/skyrover_on_rm2car/log/%Y-%m-%d %H:%M:%S_front_view.mp4", img_lst_first_person_front, fps=30)
    mediapy.write_video(f"discoverse/examples/skyrover_on_rm2car/log/%Y-%m-%d %H:%M:%S_down_view.mp4", img_lst_first_person_down, fps=30)
