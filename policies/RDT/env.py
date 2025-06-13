import importlib
import numpy as np
import time

class Env():
    def __init__(self, args):
        module = importlib.import_module(args["task_path"].replace("/", ".").replace(".py", ""))
        SimNode = getattr(module, "SimNode")
        cfg = getattr(module, "cfg")
        self.cfg = cfg
        cfg.headless = True
        self.simnode = SimNode(cfg)
        self.args = args
        self.video_list = list()

    def reset(self):
        obs = self.simnode.reset()
        self.video_list = list()
        return self.obs_ext(obs)

    def obs_ext(self, obs):
        result = dict()
        # 处理所有非图像类型的观测
        for key in self.args["obs_keys"]:
            if not key.startswith('image'):
                result[key] = np.array(obs[key])

        # 处理图像类型的观测
        image_ids = [int(key[5:]) for key in self.args["obs_keys"] if key.startswith('image')]
        img = obs['img']
        for id in image_ids:
            img_trans = np.transpose(img[id] / 255, (2, 0, 1))
            result[f'image{id}'] = img_trans

        return result

    def step(self, action):
        for _ in range(int(round(1. / self.simnode.render_fps / (self.simnode.delta_t)))):
            obs, _, _, _, _ = self.simnode.step(action)
        self.video_list.append(obs['img'])
        return self.obs_ext(obs)