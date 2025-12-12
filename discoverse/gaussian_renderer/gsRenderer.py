import os
import torch
import numpy as np
from pathlib import Path
from discoverse.gaussian_renderer import util_gau
from discoverse.gaussian_renderer.renderer_cuda import CUDARenderer
from discoverse import DISCOVERSE_ASSETS_DIR

class GSRenderer:
    def __init__(self, models_dict:dict, hf_repo_id="tatp/DISCOVERSE-models", local_dir=None):
        """
        初始化高斯飞溅渲染器
        
        Args:
            models_dict: 模型字典,键为模型名称,值为模型路径
            hf_repo_id: Hugging Face仓库ID,用于下载模型
            local_dir: 下载目标目录,默认为None(使用DISCOVERSE_ASSETS_DIR/3dgs)
        """
        self.hf_repo_id = hf_repo_id
        self.local_dir = local_dir

        self.renderer = CUDARenderer()

        self.gaussians_all:dict[util_gau.GaussianData] = {}
        self.gaussians_idx = {}
        self.gaussians_size = {}
        idx_sum = 0

        gs_model_dir = Path(os.path.join(DISCOVERSE_ASSETS_DIR, "3dgs"))

        bg_key = "background"
        if bg_key in models_dict:
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

        self.renderer.update_gaussian_data(self.gaussians_all)
        
        self.renderer.gaussian_start_indices = self.gaussians_idx
        self.renderer.gaussian_end_indices = {k: v + self.gaussians_size[k] for k, v in self.gaussians_idx.items()}
        self.renderer.gaussian_model_names = list(self.gaussians_all.keys())