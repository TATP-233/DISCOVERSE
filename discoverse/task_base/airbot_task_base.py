import os
import json
import glfw
import shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.utils import get_random_texture
from discoverse.robots_env.airbot_play_base import AirbotPlayBase
import pickle

def recoder_airbot_play(save_path, act_lst, obs_lst, cfg):
    """保存数据但延迟视频编码，返回视频编码任务信息"""
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    # 保存JSON数据
    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time" : [o['time'] for o in obs_lst],
            "obs"  : {
                "jq" : [o['jq'] for o in obs_lst],
            },
            "act"  : act_lst,
        }
        json.dump(obj, fp)

    # 保存视频数据为pickle文件，稍后编码
    video_tasks = []
    for id in cfg.obs_rgb_cam_id:
        video_data_path = os.path.join(save_path, f"cam_{id}_data.pkl")
        video_frames = [o['img'][id] for o in obs_lst]
        with open(video_data_path, 'wb') as f:
            pickle.dump(video_frames, f)
        
        video_tasks.append({
            'data_path': video_data_path,
            'output_path': os.path.join(save_path, f"cam_{id}.mp4"),
            'fps': cfg.render_set["fps"]
        })
    
    return video_tasks

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

    def random_table_height(self, table_name="table", obj_name_list=[]):
        if not hasattr(self, "table_init_posi"):
            self.table_init_posi = self.mj_model.body(table_name).pos.copy()
        change_height = np.random.uniform(-0.03, 0.07)
        self.mj_model.body(table_name).pos = self.table_init_posi.copy()
        self.mj_model.body(table_name).pos[2] = self.table_init_posi[2] - change_height
        for obj_name in obj_name_list:
            self.object_pose(obj_name)[2] -= change_height
    
    def random_table_texture(self):
        self.update_texture("tc_texture", get_random_texture())
        self.random_material("tc_texture")
    
    def random_material(self, mtl_name, random_color=False, emission=False):
        try:
            if random_color:
                self.mj_model.material(mtl_name).rgba[:3] = np.random.rand(3)
            if emission:
                self.mj_model.material(mtl_name).emission = np.random.rand()
            self.mj_model.material(mtl_name).specular = np.random.rand()
            self.mj_model.material(mtl_name).reflectance = np.random.rand()
            self.mj_model.material(mtl_name).shininess = np.random.rand()
        except KeyError:
            print(f"Warning: material {mtl_name} not found")

    def random_light(self, random_dir=True, random_color=True, random_active=True, write_color=False):
        if write_color:
            for i in range(self.mj_model.nlight):
                self.mj_model.light_ambient[i, :] = np.random.random()
                self.mj_model.light_diffuse[i, :] = np.random.random()
                self.mj_model.light_specular[i, :] = np.random.random()
        elif random_color:
            self.mj_model.light_ambient[...] = np.random.random(size=self.mj_model.light_ambient.shape)
            self.mj_model.light_diffuse[...] = np.random.random(size=self.mj_model.light_diffuse.shape)
            self.mj_model.light_specular[...] = np.random.random(size=self.mj_model.light_specular.shape)

        if random_active:
            self.mj_model.light_active[:] = np.int32(np.random.rand(self.mj_model.nlight) > 0.5).tolist()
        
        if np.sum(self.mj_model.light_active) == 0:
            self.mj_model.light_active[np.random.randint(self.mj_model.nlight)] = 1

        self.mj_model.light_pos[:,:2] = self.mj_model.light_pos0[:,:2] + np.random.normal(scale=0.3, size=self.mj_model.light_pos[:,:2].shape)
        self.mj_model.light_pos[:,2] = self.mj_model.light_pos0[:,2] + np.random.normal(scale=0.2, size=self.mj_model.light_pos[:,2].shape)

        if random_dir:
            self.mj_model.light_dir[:] = np.random.random(size=self.mj_model.light_dir.shape) - 0.5
            self.mj_model.light_dir[:,2] *= 2.0
            self.mj_model.light_dir[:] = self.mj_model.light_dir[:] / np.linalg.norm(self.mj_model.light_dir[:], axis=1, keepdims=True)
            self.mj_model.light_dir[:,2] = -np.abs(self.mj_model.light_dir[:,2])

    def domain_randomization(self):
        pass

    def checkActionDone(self):
        joint_done = np.allclose(self.sensor_joint_qpos[:6], self.target_control[:6], atol=3e-2) and np.abs(self.sensor_joint_qvel[:6]).sum() < 0.1
        gripper_done = np.allclose(self.sensor_joint_qpos[6], self.target_control[6], atol=0.016) and np.abs(self.sensor_joint_qvel[6]).sum() < 0.005
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