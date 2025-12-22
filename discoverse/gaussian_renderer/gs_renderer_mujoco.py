# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2025 Yufei Jia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .gs_renderer import GSRenderer

class GSRendererMuJoCo(GSRenderer):
    def __init__(self, models_dict: dict):
        super().__init__(models_dict)
        self.gs_idx_start = []
        self.gs_idx_end = []
        self.gs_body_ids = []

    def init_renderer(self, mj_model):
        self.gs_idx_start = []
        self.gs_idx_end = []
        self.gs_body_ids = []
        
        objects_info = []
        for i in range(mj_model.nbody):
            body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name in self.gaussian_model_names:
                start_idx = self.gaussian_start_indices[body_name]
                end_idx = self.gaussian_end_indices[body_name]
                self.gs_idx_start.append(start_idx)
                self.gs_idx_end.append(end_idx)
                self.gs_body_ids.append(i)
                objects_info.append((body_name, start_idx, end_idx))

        self.gs_idx_start = np.array(self.gs_idx_start)
        self.gs_idx_end = np.array(self.gs_idx_end)
        self.gs_body_ids = np.array(self.gs_body_ids)
        
        # Call the generic mapping method in base class
        self.set_objects_mapping(objects_info)

    def update_gaussians(self, mj_data):
        if not hasattr(self, 'gs_idx_start') or len(self.gs_idx_start) == 0:
            return

        # Batch extract position (N, 3)
        pos_values = mj_data.xpos[self.gs_body_ids]
        
        # Batch extract quaternion (N, 4) - wxyz
        quat_values = mj_data.xquat[self.gs_body_ids]
        
        # Call batch update interface
        self.update_gaussian_properties(
            pos_values,
            quat_values
        )

    def render(self, mj_model, mj_data, cam_ids, width, height, free_camera=None):
        if len(cam_ids) == 0:
            return {}, {}, {}

        # 1. Get fixed camera poses
        fixed_cam_ids = [cid for cid in cam_ids if cid != -1]
        
        if len(fixed_cam_ids) > 0:
            fixed_cam_indices = np.array(fixed_cam_ids)
            cam_pos_fixed = mj_data.cam_xpos[fixed_cam_indices]
            cam_xmat_fixed = mj_data.cam_xmat[fixed_cam_indices]
            fovy_fixed = mj_model.cam_fovy[fixed_cam_indices]
        else:
            cam_pos_fixed = np.empty((0, 3))
            cam_xmat_fixed = np.empty((0, 9))
            fovy_fixed = np.empty((0,))

        # 2. Handle free camera (cam_id == -1)
        if -1 in cam_ids:
            if free_camera is None:
                raise ValueError("free_camera must be provided if cam_id -1 is requested")
            
            # Calculate free camera pose
            camera_rmat = np.array([
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0,  1,  0],
            ])
            rotation_matrix = camera_rmat @ Rotation.from_euler('xyz', [free_camera.elevation * np.pi / 180.0, free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = free_camera.lookat + free_camera.distance * rotation_matrix[:3,2]
            
            trans = camera_position
            rmat = rotation_matrix.flatten() # (9,)
            fovy = mj_model.vis.global_.fovy
            
            cam_pos = np.vstack([cam_pos_fixed, trans])
            cam_xmat = np.vstack([cam_xmat_fixed, rmat])
            fovy_arr = np.concatenate([fovy_fixed, [fovy]])
            
            batch_indices = {cid: i for i, cid in enumerate(fixed_cam_ids)}
            batch_indices[-1] = len(fixed_cam_ids)
        else:
            cam_pos = cam_pos_fixed
            cam_xmat = cam_xmat_fixed
            fovy_arr = fovy_fixed
            batch_indices = {cid: i for i, cid in enumerate(fixed_cam_ids)}

        # Call the generic render_batch method in base class
        rgb_tensor, depth_tensor = self.render_batch(
            cam_pos,
            cam_xmat,
            height,
            width,
            fovy_arr
        )
        
        results = {}
        for cid, idx in batch_indices.items():
            results[cid] = (rgb_tensor[idx], depth_tensor[idx])
            
        return results
