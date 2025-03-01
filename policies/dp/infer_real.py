import hydra
import cv2
import time
import queue
from collections import deque
from airbot_py.airbot_play import AirbotPlay 
import numpy as np
import torch

class ImageNormalizer:
    def __init__(self):
        pass

    def normalize(self, x):
        return x * 2.0 - 1.0

    def unnormalize(self, x):
        return (x + 1.0) / 2.0


class MinMaxNormalizer:
    def __init__(self, min, range):
        self.min=min
        self.range=range

    def normalize(self, x):
        x = x.astype(np.float32)
        # nomalize to [0,1]
        nx = (x - self.min) / self.range
        # normalize to [-1, 1]
        nx = nx * 2 - 1
        return nx

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        x = nx * self.range + self.min
        return x

class MPCController:
    def __init__(self, args):
        self.args = args
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256*args.obs_steps, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(args.device)

        if args.diffusion == "ddpm":
            from cleandiffuser.diffusion.ddpm import DDPM
            x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
            x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
            self.agent = DDPM(
                nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
                optim_params={"lr": args.lr})
        elif args.diffusion == "ddim":
            from cleandiffuser.diffusion.ddim import DDIM
            self.agent = DDIM(
                nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                diffusion_steps=args.sample_steps, optim_params={"lr": args.lr})
        if args.model_path:
            self.agent.load(args.model_path)
            print(f"Load model from {args.model_path}")

        else:
            raise ValueError("Empty model for inference")            
        self.agent.model.eval()
        self.agent.model_ema.eval()
    
        # args for normalization, obtained by dataset
        # e.g.
        jq_min = [-9.2069000e-01 ,-1.9698355e+00 , 2.7621675e-01 , 1.0095695e+00 ,-1.5700147e+00 ,-2.4083598e+00 , 1.4882501e-03]   
        jq_range = [1.7821722 ,1.539348 , 2.4161224 ,1.1522678 ,0.5659977, 1.6170768 ,0.7805725]  
        action_min = [-0.9207903 ,-1.9628505 , 0.2670964 , 1.0083016 ,-1.568673 , -2.4073122 ,0.75 ]   
        action_range = [1.7822212 ,1.5348256 ,2.4343538 ,1.1552168 ,0.5657468, 1.6155746 ,1.       ]

        self.norm = {}
        self.norm['jq'] = MinMaxNormalizer(jq_min, jq_range)
        self.norm['image0'] = ImageNormalizer()
        self.norm['image1'] = ImageNormalizer()
        self.action_norm = MinMaxNormalizer(action_min, action_range)

        self.donw_sample_size = tuple(self.args.shape_meta.obs.image0.shape[1:])

    def inference(self, frame1_seq, frame2_seq, joint_state):

        if self.args.diffusion == "ddpm":
            solver = None
        elif self.args.diffusion == "ddim":
            solver = "ddim"
        elif self.args.diffusion == "dpm":
            solver = "ode_dpmpp_2"
        elif self.args.diffusion == "edm":
            solver = "euler"
        frame1_seq = [np.transpose(img / 255, (2, 0, 1)) for img in frame1_seq]
        frame2_seq = [np.transpose(img / 255, (2, 0, 1)) for img in frame2_seq]


        frame1_seq = down_sample(np.array(frame1_seq), down_sample_size)
        frame2_seq = down_sample(np.array(frame2_seq), down_sample_size)
        obs = {}
        obs['jq'] = np.array(joint_state)
        obs['image0'] = frame1_seq
        obs['image1'] = frame2_seq
        condition = {}
        for k in obs.keys():
            obs_seq = obs[k].astype(np.float32)  # (obs_steps, obs_dim)
            nobs = self.norm[k].normalize(obs_seq)
            nobs = torch.tensor(nobs, device=self.args.device, dtype=torch.float32)  # (obs_steps, obs_dim)
            nobs = nobs[None, :].expand(self.args.num_envs, *nobs.shape) # torch.Size([num_envs, obs_steps, obs_dim])
            condition[k] = nobs  #jq,image0,image1
        
        # predict
        with torch.no_grad():
            prior = torch.zeros((self.args.num_envs, self.args.horizon, self.args.action_dim), device=self.args.device)
            naction, _ = self.agent.sample(prior=prior, n_samples=self.args.num_envs, sample_steps=self.args.eval_sample_steps,
                                    solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)

        print(condition['image0'].shape)
        # unnormalize prediction
        naction = naction.detach().to('cpu').numpy() 
        action_pred = self.action_norm.unnormalize(naction)  

        start = self.args.obs_steps - 1
        end = start + self.args.action_steps
        action = np.squeeze(action_pred[:, start:end, :]) 
        return action

def down_sample(img, size):
    b, c, h, w = img.shape
    downsampled = np.zeros((b, c, size[0], size[1]), dtype=img.dtype)

    for i in range(b):
        # (C,H,W) -> (H,W,C)
        frame = img[i].transpose(1, 2, 0)
        resized = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        downsampled[i] = resized.transpose(2, 0, 1)

    return downsampled

@hydra.main(version_base=None, config_path="./2real/configs/dp", config_name="laptop_close")
def main(args):
    
    # Initialize airbot instance
    airbot_play = AirbotPlay()
    
    # Initialize position
    joints = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599]
    eef = 0.5
    airbot_play.set_target_joint_q(joints)
    airbot_play.set_target_end(eef)

    # Initialize cameras
    cap1 = cv2.VideoCapture(args.global_camid)
    cap2 = cv2.VideoCapture(args.wrist_camid)

    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 30.0

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter('cam0.mp4', fourcc, fps, (frame_width1, frame_height1))
    out2 = cv2.VideoWriter('cam1.mp4', fourcc, fps, (frame_width2, frame_height2))

    # Initialize MPC
    mpc = MPCController(args)

    # Initialize data buffers
    frame1_buffer = deque(maxlen=args.obs_steps)
    frame2_buffer = deque(maxlen=args.obs_steps)
    joint_buffer = deque(maxlen=args.obs_steps)


    try:
        for i in range(args.eval_episodes):
            # Reset position
            joints = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599]
            eef = 0.5
            airbot_play.set_target_joint_q(joints)
            airbot_play.set_target_end(eef)

            # init
            # wait for some times to fill the buffers with data caught in initial position
            wait_time = 0
            while True:
                wait_time += 1
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                arm = airbot_play.get_current_joint_q()
                eef = airbot_play.get_current_end()
                jq = np.concatenate([arm, [eef]])

                cv2.imshow('image1',frame1)
                cv2.imshow('image2', frame2)
                key = cv2.waitKey(1)

                if key == ord(" ") and wait_time > 20 and ret1 and ret2 and jq is not None:
                    frame1_buffer.extend(np.tile(frame1, (args.obs_steps, 1, 1, 1)))
                    frame2_buffer.extend(np.tile(frame2, (args.obs_steps, 1, 1, 1)))
                    joint_buffer.extend(np.tile(jq, (args.obs_steps, 1)))
                    break

            new_round = False
            while True:
                if new_round:
                    break
                action = mpc.inference(frame1_buffer, frame2_buffer, joint_buffer)
                if action is not None:
                    # Excute each action step in action sequence
                    for act in action:
                        if new_round:
                            break
                        # airbot_play.set_target_joint_q(act[:6], blocking=True, vel=0.2, acceleration=0.2, use_planning=False)
                        airbot_play.set_target_joint_q(act[:6], blocking=False, vel=0.2, acceleration=0.2, use_planning=False)
                        airbot_play.set_target_end(act[-1], blocking=False)

                        # Wait for new frame data and update the buffers
                        while True:
                            if new_round:
                                break
                            ret1, frame1 = cap1.read()
                            ret2, frame2 = cap2.read()
                            cv2.imshow('image1',frame1)
                            cv2.imshow('image2', frame2)
                            key = cv2.waitKey(1)
                            if key == ord("c"):
                                print("-" * 100 + f"new_round_{i:03d} / 50")
                                new_round = True
                            elif key == ord("q"):
                                raise KeyboardInterrupt

                            arm = airbot_play.get_current_joint_q()
                            eef = airbot_play.get_current_end()
                            jq = np.concatenate([arm, [eef]])
                            if ret1 and ret2 and jq is not None:
                                frame1_buffer.append(frame1)
                                frame2_buffer.append(frame2)
                                joint_buffer.append(jq)

                                # Write video
                                out1.write(frame1)
                                out2.write(frame2)
                                break

    except KeyboardInterrupt:
        print("Stopping system...")
        cap1.release()
        cap2.release()
        out1.release()  # Release VideoWriter
        out2.release()  

    finally:
        cv2.destroyAllWindows()
        # reset posiotion
        joints = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599]
        eef = 0.5
        airbot_play.set_target_joint_q(joints)
        airbot_play.set_target_end(eef)
        airbot_play.shutdown()

if __name__ == "__main__":
    main()