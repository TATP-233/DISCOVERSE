import sys
sys.path.append('./') 
sys.path.insert(0, './policies/RDT')
from model import *

import os
from pathlib import Path

import numpy as np
import yaml
from datetime import datetime
from argparse import ArgumentParser

from env import Env
import datetime

# Interpolate the actions to make the robot move smoothly
def interpolate_action(prev_action, cur_action):
    steps = 0.05 * np.ones(len(cur_action)) #The maximum change allowed for each joint per timestep
    # steps = np.concatenate((np.array(arm_steps_length), np.array(arm_steps_length)), axis=0)
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

            # 获取观测：包括 head/left/right 摄像头的图像与 joint 状态
            cam_obs = [obs["image0"], obs["image1"], obs["image1"]]  # BGR to RGB 若需要
            agent_pos = obs["jq"]
            
            model.update_observation_window(cam_obs, agent_pos)
            actions = model.get_action()  # shape: (64, 7)

            for action in actions:
                # 动作插值，使动作更平滑
                if t!=0:
                    interp_actions = interpolate_action(last_action, action)
                else:
                    interp_actions = action[np.newaxis, :]
                for interp_action in interp_actions:
                    # 当前执行的动作
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
                break

        print("success rate:", success_sum, "/", i+1, "=", success_sum / (i+1))

        # save video
        import mediapy
        for id in env.cfg.obs_rgb_cam_id:
            mediapy.write_video(os.path.join(save_dir, f"{i}_cam_{id}.mp4"), [videos[id] for videos in env.video_list], fps=env.cfg.render_set["fps"])

def main(usr_args):
    task_name = usr_args.task_name
    model_name = usr_args.model_name
    checkpoint_num = usr_args.checkpoint_num

    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    env = Env(args)

    args['model_name'] = model_name
    args['checkpoint_id'] = checkpoint_num

    # if running on pretrained pipe line or run on single gpu, deepspeed will not save mp_rank_00_model_states.pt
    try:
        rdt = RDT(f"./checkpoints/{model_name}/checkpoint-{checkpoint_num}/pytorch_model/mp_rank_00_model_states.pt", task_name)
    except:
        rdt = RDT(f"./checkpoints/{model_name}/checkpoint-{checkpoint_num}/", task_name)
    rdt.random_set_language()

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M")
    save_dir = Path(f'eval_result/{task_name}/{current_time}')
    save_dir.mkdir(parents=True, exist_ok=True)

    eval(args, env, rdt, save_dir)

if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()
    
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str, default='block_hammer_beat')
    parser.add_argument('model_name', type=str)
    parser.add_argument('checkpoint_num', type=int, default=1000)
    usr_args = parser.parse_args()
    
    main(usr_args)
