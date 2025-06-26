import sys
sys.path.append('./')
sys.path.insert(0, './policies/openpi')
from pi_model import *

import os
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import yaml
import numpy as np

from env import Env

# 动作插值
def interpolate_action(prev_action, cur_action):
    steps = 0.05 * np.ones(len(cur_action))
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def eval(args, env, model, save_dir):
    success_sum = 0

    for i in range(args["eval_num"]):
        obs = env.reset()
        t = 0
        success = False

        while t < args["max_steps"]:
            print("step:", t, "/", args["max_steps"])

            cam_obs = [obs["image0"], obs["image1"], obs["image1"]]  # 如果需要 BGR->RGB 可加 .[..., ::-1]
            agent_pos = obs["jq"]

            model.update_observation_window(cam_obs, agent_pos)
            actions = model.get_action()  # shape: (N, D)

            for action in actions:
                if t != 0:
                    interp_actions = interpolate_action(last_action, action)
                else:
                    interp_actions = action[np.newaxis, :]
                for interp_action in interp_actions:
                    obs = env.step(np.array(interp_action))
                    last_action = interp_action.copy()
                t += 1

            if obs.get("success", False):
                success = True
                break

            if t >= args["max_steps"]:
                break

        if success:
            success_sum += 1

        print("success rate:", success_sum, "/", i+1, "=", success_sum / (i+1))

        # 保存视频
        import mediapy
        for id in env.cfg.obs_rgb_cam_id:
            mediapy.write_video(
                os.path.join(save_dir, f"{i}_cam_{id}.mp4"),
                [videos[id] for videos in env.video_list],
                fps=env.cfg.render_set["fps"]
            )

def main(usr_args):
    task_name = usr_args.task_name
    train_config_name = usr_args.train_config_name
    model_name = usr_args.model_name
    checkpoint_num = usr_args.checkpoint_num

    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    env = Env(args)

    args['model_name'] = model_name
    args['checkpoint_id'] = checkpoint_num

    # 加载模型
    model = PI0(task_name, train_config_name, model_name, checkpoint_num)
    model.random_set_language()

    # 保存目录
    current_time = datetime.now().strftime("%y%m%d%H%M")
    save_dir = Path(f'eval_result_pi0/{train_config_name}_{task_name}/{current_time}')
    save_dir.mkdir(parents=True, exist_ok=True)

    eval(args, env, model, save_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str)
    parser.add_argument('train_config_name', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('checkpoint_num', type=int)
    usr_args = parser.parse_args()

    main(usr_args)
