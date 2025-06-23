import os
import json
import fractions
import av.video
import glfw
import shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg
from collections import defaultdict
import av


def recoder_airbot_play(save_path, act_lst, obs_lst, cfg: AirbotPlayCfg):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        time = []
        jq = []
        images = defaultdict(list)
        for obs in obs_lst:
            time.append(obs['time'])
            jq.append(obs['jq'])
            for img_id in cfg.obs_rgb_cam_id:
                images[img_id].append(obs['img'][img_id])
        json.dump({
            "time" : time,
            "obs"  : {
                "jq" : jq,},
            "act"  : act_lst,
        }, fp)

    for id, image_list in images.items():
        container = av.open(os.path.join(save_path, f"cam_{id}.mp4"), "w", format="mp4")
        stream: av.video.stream.VideoStream = container.add_stream("h264", options={"preset": "fast"})
        stream.width = cfg.render_set["width"]
        stream.height = cfg.render_set["height"]
        stream.pix_fmt = "yuv420p"
        stream.time_base = fractions.Fraction(1, int(1e9))
        start_time = time[0]
        last_time = 0
        container.metadata["comment"] = str({"base_stamp": int(start_time * 1e9)})
        for index, img in enumerate(image_list):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            cur_time = time[index]
            frame.pts = int((cur_time - start_time) * 1e9)
            frame.time_base = stream.time_base
            assert cur_time > last_time, f"Time error: {cur_time} <= {last_time}"
            last_time = cur_time
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()


class AirbotPlayTaskBase(AirbotPlayBase):
    target_control = np.zeros(7)
    joint_move_ratio = np.zeros(7)
    action_done_dict = {
        "joint"   : False,
        "gripper" : False,
        "delay"   : False,
    }
    delay_cnt = 0
    reset_sig = False
    cam_id = 0

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def domain_randomization(self):
        pass

    def checkActionDone(self):
        joint_done = np.allclose(self.sensor_joint_qpos[:6], self.target_control[:6], atol=3e-2) and np.abs(self.sensor_joint_qvel[:6]).sum() < 0.1
        gripper_done = np.allclose(self.sensor_joint_qpos[6], self.target_control[6], atol=0.4) and np.abs(self.sensor_joint_qvel[6]).sum() < 0.125
        self.delay_cnt -= 1
        delay_done = (self.delay_cnt<=0)
        self.action_done_dict = {
            "joint"   : joint_done,
            "gripper" : gripper_done,
            "delay"   : delay_done,
        }
        return joint_done and gripper_done and delay_done

    def printMessage(self):
        super().printMessage()
        print("    target control = ", self.target_control)
        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

        print("camera foyv = ", self.mj_model.vis.global_.fovy)
        cam_xyz, cam_wxyz = self.getCameraPose(self.cam_id)
        print(f"    camera_{self.cam_id} =\n({cam_xyz}\n{Rotation.from_quat(cam_wxyz[[1,2,3,0]]).as_matrix()})")

    def check_success(self):
        raise NotImplementedError
    
    def on_key(self, window, key, scancode, action, mods):
        ret = super().on_key(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            if key == glfw.KEY_MINUS:
                self.mj_model.vis.global_.fovy = np.clip(self.mj_model.vis.global_.fovy*0.95, 5, 175)
            elif key == glfw.KEY_EQUAL:
                self.mj_model.vis.global_.fovy = np.clip(self.mj_model.vis.global_.fovy*1.05, 5, 175)
        return ret