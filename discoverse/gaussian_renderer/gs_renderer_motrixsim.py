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
from typing import Tuple, List, Union, Dict, Optional, Any, TYPE_CHECKING

import numpy as np
from torch import Tensor

import mujoco

if TYPE_CHECKING:
    # only import motrixsim for type checking to avoid hard dependency at runtime
    import motrixsim

from .gs_renderer_mujoco import GSRendererMuJoCo

class MjMxBridge:
    def __init__(self, mjcf_path: str, assets: Optional[Dict[str, Any]] = None):
        if assets:
            self._mj_model = mujoco.MjModel.from_xml_path(mjcf_path, assets=assets)
        else:
            self._mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        self.map_qpos_idx_mjmx = np.arange(self._mj_model.nq, dtype=np.int32)
        for i in range(self._mj_model.nbody):
            body = self._mj_model.body(i)
            if body.dofnum == 6:
                # xyz + quat[wxyz] -> xyz + quat[xyzw]
                jnt_adr = body.jntadr[0]
                self.map_qpos_idx_mjmx[jnt_adr+3:jnt_adr+7] = [
                    jnt_adr+4, jnt_adr+5, jnt_adr+6, jnt_adr+3
                ]

    @property
    def data(self) -> mujoco.MjData:
        return self._mj_data

    def update(self, mx_data: "motrixsim.SceneData") -> mujoco.MjData:
        assert mx_data.dof_pos.shape[0] == self._mj_model.nq, "DOF position size mismatch"
        self._mj_data.qpos[self.map_qpos_idx_mjmx] = mx_data.dof_pos[:]
        mujoco.mj_forward(self._mj_model, self._mj_data)
        return self._mj_data

    def load_keyframe(self, mx_data: "motrixsim.SceneData", mx_model: "motrixsim.SceneModel", keyframe_idx: Union[int, str]) -> None:
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, self._mj_model.key(keyframe_idx).id)
        mx_data.set_dof_pos(self._mj_data.qpos[self.map_qpos_idx_mjmx], mx_model)
        mx_data.set_dof_vel(self._mj_data.qvel)
        mx_data.actuator_ctrls = self._mj_data.ctrl.copy()

class GSRendererMotrixSim(GSRendererMuJoCo):
    def __init__(self, models_dict: Dict[str, str], bridge:Optional[MjMxBridge] = None, mjcf_path: Optional[str]=None, assets: Optional[Dict[str, Any]] = None):
        if bridge is not None:
            self.bridge = bridge
        elif mjcf_path is not None:
            self.bridge = MjMxBridge(mjcf_path=mjcf_path, assets=assets)
        else:
            raise ValueError("Either bridge or mjcf_path must be provided.")
        super().__init__(models_dict)
        super().init_renderer(self.bridge._mj_model)

    def init_renderer(self, mj_model=None) -> None:
        # hold-place Initialization is done in the constructor
        pass

    def update_gaussians(self, mx_data: "motrixsim.SceneData") -> None:
        mj_data = self.bridge.update(mx_data)
        super().update_gaussians(mj_data)

    def render(self, 
               cam_ids:Union[List[int], np.ndarray], 
               width:int, height:int) -> Dict[int, Tuple[Tensor, Tensor]]:
        if -1 in cam_ids:
            raise NotImplementedError("Free camera rendering not supported in MotrixSim bridge.")
        return super().render(self.bridge._mj_model, self.bridge._mj_data, cam_ids, width, height)