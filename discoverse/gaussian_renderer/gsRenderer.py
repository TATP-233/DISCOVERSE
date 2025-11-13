import os
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from discoverse.gaussian_renderer import util_gau
from discoverse.gaussian_renderer.renderer_cuda import CUDARenderer

from discoverse import DISCOVERSE_ASSETS_DIR

class GSRenderer:
    def __init__(self, models_dict:dict, render_width=1920, render_height=1080, hf_repo_id="tatp/DISCOVERSE-models", local_dir=None):
        """
        初始化高斯飞溅渲染器
        
        Args:
            models_dict: 模型字典，键为模型名称，值为模型路径
            render_width: 渲染宽度
            render_height: 渲染高度
            hf_repo_id: Hugging Face仓库ID，用于下载模型
            local_dir: 下载目标目录，默认为None（使用DISCOVERSE_ASSETS_DIR/3dgs）
        """
        self.width = render_width
        self.height = render_height
        self.hf_repo_id = hf_repo_id
        self.local_dir = local_dir

        self.camera = util_gau.Camera(self.height, self.width)

        self.update_gauss_data = False

        self.scale_modifier = 1.

        self.renderer = CUDARenderer(self.camera.w, self.camera.h)
        self.camera_tran = np.zeros(3)
        self.camera_quat = np.zeros(4)

        self.gaussians_all:dict[util_gau.GaussianData] = {}
        self.gaussians_idx = {}
        self.gaussians_size = {}
        idx_sum = 0

        gs_model_dir = Path(os.path.join(DISCOVERSE_ASSETS_DIR, "3dgs"))

        bg_key = "background"
        data_path = Path(os.path.join(gs_model_dir, models_dict[bg_key]))
        gs = util_gau.load_ply(data_path, hf_repo_id=self.hf_repo_id, local_dir=self.local_dir)
        if "background_env" in models_dict.keys():
            bgenv_key = "background_env"
            bgenv_gs = util_gau.load_ply(
                Path(os.path.join(gs_model_dir, models_dict[bgenv_key])),
                hf_repo_id=self.hf_repo_id,
                local_dir=self.local_dir
            )
            gs.xyz = np.concatenate([gs.xyz, bgenv_gs.xyz], axis=0)
            gs.rot = np.concatenate([gs.rot, bgenv_gs.rot], axis=0)
            gs.scale = np.concatenate([gs.scale, bgenv_gs.scale], axis=0)
            gs.opacity = np.concatenate([gs.opacity, bgenv_gs.opacity], axis=0)
            gs.sh = np.concatenate([gs.sh, bgenv_gs.sh], axis=0)

        self.gaussians_all[bg_key] = gs
        self.gaussians_idx[bg_key] = idx_sum
        self.gaussians_size[bg_key] = gs.xyz.shape[0]
        idx_sum = self.gaussians_size[bg_key]

        for i, (k, v) in enumerate(models_dict.items()):
            if k != "background" and k != "background_env":
                data_path = Path(os.path.join(gs_model_dir, v))
                gs = util_gau.load_ply(data_path, hf_repo_id=self.hf_repo_id, local_dir=self.local_dir)
                self.gaussians_all[k] = gs
                self.gaussians_idx[k] = idx_sum
                self.gaussians_size[k] = gs.xyz.shape[0]
                idx_sum += self.gaussians_size[k]

        self.update_activated_renderer_state(self.gaussians_all)

        for name in self.gaussians_all.keys():
            # :TODO: 找到哪里被改成torch了
            try:
                self.gaussians_all[name].R = self.gaussians_all[name].R.numpy()
            except:
                pass

    def update_camera_intrin_lazy(self):
        if self.camera.is_intrin_dirty:
            self.renderer.update_camera_intrin(self.camera)
            self.camera.is_intrin_dirty = False

    def update_activated_renderer_state(self, gaus: util_gau.GaussianData):
        self.renderer.update_gaussian_data(gaus)
        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.update_camera_pose(self.camera)
        self.renderer.update_camera_intrin(self.camera)
        self.renderer.set_render_reso(self.camera.w, self.camera.h)

    def set_obj_pose(self, obj_name, trans, quat_wzyx):
        if not ((self.gaussians_all[obj_name].origin_rot == quat_wzyx).all() and (self.gaussians_all[obj_name].origin_xyz == trans).all()):
            self.update_gauss_data = True
            self.gaussians_all[obj_name].origin_rot = quat_wzyx.copy()
            self.gaussians_all[obj_name].origin_xyz = trans.copy()
            self.renderer.gau_xyz_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(trans).cuda().requires_grad_(False)
            self.renderer.gau_rot_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(quat_wzyx).cuda().requires_grad_(False)

    def set_camera_pose(self, trans, quat_xyzw):
        if not ((self.camera_tran == trans).all() and (self.camera_quat == quat_xyzw).all()):
            self.camera_tran[:] = trans[:]
            self.camera_quat[:] = quat_xyzw[:]
            rmat = Rotation.from_quat(quat_xyzw).as_matrix()
            self.renderer.update_camera_pose_from_topic(self.camera, rmat, trans)

    def set_camera_fovy(self, fovy):
        if not fovy == self.camera.fovy:
            self.camera.update_fovy(fovy)
    
    def set_camera_resolution(self, height, width):
        if not (height == self.camera.h and width == self.camera.w):
            self.camera.update_resolution(height, width)
            self.renderer.set_render_reso(width, height)

    def render(self, render_depth=False):
        self.update_camera_intrin_lazy()
        return self.renderer.draw(render_depth)
