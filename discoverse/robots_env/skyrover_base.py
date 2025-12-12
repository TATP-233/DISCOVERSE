import numpy as np

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class SkyRoverCfg(BaseConfig):
    mjcf_file_path = "mjcf/skyrover_floor.xml"
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080 
    }
    obs_rgb_cam_id   = None
    obs_depth_cam_id = None

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
        
    }

class SkyRoverBase(SimulatorBase):
    def updateControl(self, action):
        self.mj_data.ctrl[:] = action[:]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq"  : self.mj_data.qpos.tolist(),
            "jv"  : self.mj_data.qvel.tolist(),
            "img" : self.img_rgb_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":

    cfg = SkyRoverCfg()
    cfg.use_gaussian_renderer = True

    cfg.gs_model_dict["background"] = "scene/riverside_aligned/point_cloud_high.ply"

    exec_node = SkyRoverBase(cfg)
    obs = exec_node.reset()
    print(obs)


    cnt = 0
    action = np.zeros(15)
    while exec_node.running:
        if cnt * exec_node.delta_t < 0.5:
            action[0] = 0
            action[1] = 80
            action[6] = 80
        elif cnt * exec_node.delta_t < 1.:
            action[0] = -545
            action[1] = 0
            action[6] = 0
        else:
            cnt = 0
        cnt += 1

        obs, pri_obs, rew, ter, info = exec_node.step(action)
