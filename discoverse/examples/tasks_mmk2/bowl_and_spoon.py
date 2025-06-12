from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2
import numpy as np


class SimNode(MMK2TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.arm_action = "pick"

    def domain_randomization(self):
        pass

    def check_success(self):
        pass


cfg = MMK2Cfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "pick"


cfg.mjcf_file_path = "mjcf/tasks_mmk2/bowl_and_spoon.xml"
cfg.sync     = True  #加速
cfg.headless = False
cfg.render_set  = {
    "fps"    : 25,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0,1,2]
cfg.save_mjb_and_task_config = True

