import importlib
import numpy as np


class Env:
    def __init__(self, args):
        module = importlib.import_module(
            args.task_path.replace("/", ".").replace(".py", "")
        )
        SimNode = getattr(module, "SimNode")  # SimNode
        cfg = getattr(module, "cfg")  # cfg
        self.cfg = cfg
        cfg.headless = True
        self.simnode = SimNode(cfg)
        self.args = args
        self.obs_steps = args.obs_steps
        self.obs_que = None
        self.video_list = list()

    def reset(self):
        obs, t = self.simnode.reset(), 0
        self.video_list = list()
        from collections import deque

        self.obs_que = deque([obs], maxlen=self.obs_steps + 1)
        return self.obs_que_ext(), t

    def obs_que_ext(self):
        result = dict()
        # 处理所有非图像类型的观测
        for key in self.args.obs_keys:
            if not key.startswith("image"):
                result[key] = self.stack_last_n_obs(
                    [np.array(obs[key]) for obs in self.obs_que]
                )
        # 处理图像类型的观测
        image_ids = [
            int(key[5:]) for key in self.args.obs_keys if key.startswith("image")
        ]
        imgs = {id: [] for id in image_ids}
        for obs in self.obs_que:
            img = obs["img"]
            for id in image_ids:
                img_trans = np.transpose(img[id] / 255, (2, 0, 1))  # 转置并归一化
                imgs[id].append(img_trans)

        for id in image_ids:
            result[f"image{id}"] = self.stack_last_n_obs(imgs[id])
        return result

    def step(self, action):
        success = 0
        for act in action:  # 依次执行每个动作
            for _ in range(
                int(round(1.0 / self.simnode.render_fps / (self.simnode.delta_t)))
            ):
                obs, _, _, _, _ = self.simnode.step(act)
            self.obs_que.append(obs)  # 添加单个obs
            self.video_list.append(obs["img"])
            if self.simnode.check_success():
                success = 1
                break
        return self.obs_que_ext(), success

    def stack_last_n_obs(self, all_obs):
        assert len(all_obs) > 0
        result = np.zeros(
            (self.obs_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
        )
        start_idx = -min(self.obs_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if self.obs_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
        return result
