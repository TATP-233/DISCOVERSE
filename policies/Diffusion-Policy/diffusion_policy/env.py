import importlib
import numpy as np

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
        self.last_qpos = np.zeros(19)
        self.last_action = np.zeros(19)
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
    
    # Interpolate the actions to make the robot move smoothly
    def interpolate_action(self, prev_action, cur_action):
        steps = 0.01 * np.ones(len(cur_action)) #The maximum change allowed for each joint per timestep
        diff = np.abs(cur_action - prev_action)
        step = np.ceil(diff / steps).astype(int)
        step = np.max(step)
        if step <= 1:
            return cur_action[np.newaxis, :]
        new_actions = np.linspace(prev_action, cur_action, step + 1)
        return new_actions[1:]

    def step(self, actions):
        success = 0
        num = 0
        
        for action in actions:  # 依次执行每个动作
            num += 1
            # 动作插值，使动作更平滑
            if not self.first_step:
                interp_actions = self.interpolate_action(self.last_action, action)
            else:
                self.first_step = False
                interp_actions = action[np.newaxis, :]
            for interp_action in interp_actions:
                # 当前执行的动作
                obs, _, _, _, _ = self.simnode.step(interp_action)
                self.last_action = interp_action.copy()

            self.video_list.append(obs['img'])
            if self.simnode.check_success():
                success = 1
                break
        return self.obs_ext(obs), success, num