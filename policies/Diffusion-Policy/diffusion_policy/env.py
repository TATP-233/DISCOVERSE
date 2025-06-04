import importlib
import numpy as np

class Env():
    def __init__(self, args):
        module = importlib.import_module(args["task_path"].replace("/", ".").replace(".py", ""))
        SimNode = getattr(module, "SimNode")
        cfg = getattr(module, "cfg")
        self.cfg = cfg
        cfg.headless = False
        self.simnode = SimNode(cfg)
        self.args = args
        self.video_list = list()
        self.last_qpos = np.zeros(19)
        self.first_step = True

    def reset(self):
        obs, t = self.simnode.reset(), 0
        self.video_list = list()
        return self.obs_ext(obs), t

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
    
    def match_euclidean(self, q_cur, q_seq):
        dists = np.linalg.norm(q_seq - q_cur, axis=1)
        best_idx = np.argmin(dists)
        return best_idx + 1

    def step(self, action):
        success = 0
        num = 0
        # 动作状态匹配，从最接近的状态开始
        # if self.first_step:
        #     self.first_step = False
        # else:
        #     start_index = min(self.match_euclidean(self.last_qpos, action), action.shape[0]-1) # start_index要小于action长度
        #     action = action[start_index:]

        for act in action:  # 依次执行每个动作
            num += 1
            for _ in range(int(round(1. / self.simnode.render_fps / (self.simnode.delta_t)))):
                obs, _, _, _, _ = self.simnode.step(act)
            self.last_qpos = obs['jq']
            self.video_list.append(obs['img'])
            if self.simnode.check_success():
                success = 1
                break
        return self.obs_ext(obs), success, num