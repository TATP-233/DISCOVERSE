import glfw
import mujoco
import numpy as np
from discoverse.robots_env.leaphand_sensor_env_base import LeapHandCfg, LeapHandBase 

action = np.zeros(16)
obs_lst = []

class SimNode(LeapHandBase):
    key_id = 0
    target_action = np.zeros(16)
    joint_move_ratio = np.zeros(16)

    def __init__(self, cfg):
        super().__init__(cfg)

    def resetState(self):
        super().resetState()
        global action, obs_lst
        obs_lst.clear()
        self.key_id = 0
        self.update_control_from_keyframe("0")
        action[:] = self.target_action[:]

    def update_control_from_keyframe(self, key):
        self.target_action = self.mj_model.key(key).qpos[:self.nj].copy()
        global action
        dif = np.abs(action - self.target_action)
        self.joint_move_ratio = dif / (np.max(dif) + 1e-6)

    def on_key(self, window, key, scancode, action, mods):
        super().on_key(window, key, scancode, action, mods)
        # press space to switch to 12 keyframes
        if action == glfw.PRESS:
            print("key:", key)
            if key == glfw.KEY_SPACE:
                self.key_id += 1
                if self.key_id > 12:
                    self.key_id = 12
                    print("key_id is out of range")
                self.update_control_from_keyframe(self.key_id)
    
    def getObservation(self):
        self.obs = {
            # "jq"    : self.mj_data.qpos[:self.nj].tolist(),
            # "jv"    : self.mj_data.qvel[:self.nj].tolist(),
            "img"   : self.img_rgb_obs_s,
            "thumb" : self.mj_data.sensordata[0:32].tolist(),
            "index" : self.mj_data.sensordata[32:64].tolist(),
            "mid"   : self.mj_data.sensordata[64:96].tolist(),
            "ring"  : self.mj_data.sensordata[96:128].tolist()
        }
        return self.obs

    def get_sensor_data(self):
        self.sensor_data={
            "thumb": self.mj_data.sensordata[0:32].tolist(),
            "index" : self.mj_data.sensordata[32:64].tolist(),
            "mid" : self.mj_data.sensordata[64:96].tolist(),
            "ring" : self.mj_data.sensordata[96:128].tolist()
        }
        return self.sensor_data

cfg = LeapHandCfg()
cfg.use_gaussian_renderer = False

cfg.timestep     = 0.001
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.init_key     = "10"
cfg.render_set   = {
    "fps"    : 30,
    "width"  : 1920, # 640,
    "height" : 1080  # 480
}
cfg.obs_rgb_cam_id   = -1

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    def step_func(current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

    sim_node = SimNode(cfg)
    sim_node.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False


    try:
        while sim_node.running:

            for i in range(sim_node.nj):
                action[i] = step_func(action[i], sim_node.target_action[i], 2. * sim_node.joint_move_ratio[i] * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

            obs, _, _, _, _ = sim_node.step(action)
            if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                obs_lst.append(obs)

            #  4 Sensor Arrays Visualization
            thumb_array=np.array(sim_node.get_sensor_data()["thumb"]).reshape((8,4))
            index_array=np.array(sim_node.get_sensor_data()["index"]).reshape((8,4))
            mid_array=np.array(sim_node.get_sensor_data()["mid"]).reshape((8,4))
            ring_array=np.array(sim_node.get_sensor_data()["ring"]).reshape((8,4))
            # print(thumb_array)
            # print("\n")

            # max_threshold=100.0
            # thumb_array=np.clip(thumb_array, amin=None, a_max=max_threshold)
            # index_array=np.clip(index_array, amin=None, a_max=max_threshold)
            # mid_array=np.clip(mid_array, amin=None, a_max=max_threshold)
            # ring_array=np.clip(ring_array, amin=None, a_max=max_threshold)        
    except Exception as e:
        print(e)
    finally:
        if len(obs_lst):
            import json
            import mediapy
            mediapy.write_video("leaphand.mp4", [o["img"][-1] for o in obs_lst], fps=cfg.render_set["fps"])

            for fn in ["thumb", "index", "mid", "ring"]:
                with open(f"{fn}.json", "w") as f:
                    json.dump([o[f"{fn}"] for o in obs_lst], f)

            # with open("leadphand.json", "w") as f:
                # json.dump([o["fingers"] for o in obs_lst], f)
            # print("data saved")

        else:
            print("no data")

    # with open("leadphand.json", "r") as f:
    #     finger_dict = json.load(f)
    #     print(finger_dict.shape)
    #     print(type(finger_dict))

