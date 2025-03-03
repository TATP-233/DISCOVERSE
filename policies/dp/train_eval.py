import hydra
import os
import warnings

warnings.filterwarnings("ignore")

import pathlib
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from policies.dp.utils import set_seed, Logger, report_parameters

from policies.dp.diffuser.dataset.dp_dataset import DPDataset
from policies.dp.diffuser.dataset.dataset_utils import loop_dataloader

from policies.dp.env import Env


def eval(args, env, dataset, agent, gradient_step):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_steps = []
    episode_success = []

    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"

    for i in range(args.eval_episodes):
        obs, t = env.reset()  # {obs_name: (obs_steps, obs_dim)}
        success = 0

        while t < args.max_episode_steps:
            # 接收n个obs
            condition = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)  # (obs_steps, obs_dim)
                nobs = dataset.normalizer["obs"][k].normalize(obs_seq)
                nobs = torch.tensor(
                    nobs, device=args.device, dtype=torch.float32
                )  # (obs_steps, obs_dim)
                nobs = nobs[None, :].expand(
                    1, *nobs.shape
                )  # torch.Size([1, obs_steps, obs_dim])
                condition[k] = nobs
            # predict
            with torch.no_grad():
                prior = torch.zeros(
                    (1, args.horizon, args.action_dim), device=args.device
                )
                naction, _ = agent.sample(
                    prior=prior,
                    n_samples=1,
                    sample_steps=args.sample_steps,
                    solver=solver,
                    condition_cfg=condition,
                    w_cfg=1.0,
                    use_ema=True,
                )

            # unnormalize prediction
            naction = (
                naction.detach().to("cpu").numpy()
            )  # (1,horizon, action_dim) dim=0在训练时是Batchsize，在推理时是env_num
            action_pred = dataset.normalizer["action"].unnormalize(naction)
            # get action
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = np.squeeze(action_pred[:, start:end, :], axis=0)  # 多一个env_num维度
            obs, success = env.step(action)
            t += args.action_steps

            if success:
                break

        import mediapy

        for id in env.cfg.obs_rgb_cam_id:
            mediapy.write_video(
                os.path.join(args.work_dir, f"videos/{gradient_step}_{i}_cam_{id}.mp4"),
                [videos[id] for videos in env.video_list],
                fps=env.cfg.render_set["fps"],
            )
        print(f"[Episode {1+i}] success:{success}")
        episode_steps.append(t)
        episode_success.append(success) if success == 1 else episode_success.append(0)
    print(
        f"Mean step: {np.nanmean(episode_steps)} Mean success: {np.nanmean(episode_success)}"
    )
    return {
        "mean_step": np.nanmean(episode_steps),
        "mean_success": np.nanmean(episode_success),
    }


from omegaconf import DictConfig


@hydra.main(config_path="./configs", config_name="block_place")
def main(args: DictConfig):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    env = Env(args)
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = DPDataset(
        dataset_path,
        horizon=args.horizon,
        obs_keys=args.obs_keys,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        abs_action=args.abs_action,
    )
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "dit":
        from policies.dp.diffuser.nn_condition import MultiImageObsCondition
        from policies.dp.diffuser.nn_diffusion import DiT1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta,
            emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
        ).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim,
            emb_dim=256 * args.obs_steps,
            d_model=320,
            n_heads=10,
            depth=2,
            timestep_emb_type="fourier",
        ).to(args.device)

    elif args.nn == "chi_unet":
        from policies.dp.diffuser.nn_condition import MultiImageObsCondition
        from policies.dp.diffuser.nn_diffusion import ChiUNet1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta,
            emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
        ).to(args.device)
        nn_diffusion = ChiUNet1d(
            args.action_dim,
            256,
            args.obs_steps,
            model_dim=256,
            emb_dim=256,
            dim_mult=[1, 2, 2],
            obs_as_global_cond=True,
            timestep_emb_type="positional",
        ).to(args.device)

    elif args.nn == "chi_transformer":
        from policies.dp.diffuser.nn_condition import MultiImageObsCondition
        from policies.dp.diffuser.nn_diffusion import ChiTransformer

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta,
            emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
            keep_horizon_dims=True,
        ).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim,
            256,
            args.horizon,
            args.obs_steps,
            d_model=256,
            nhead=4,
            num_layers=4,
            timestep_emb_type="positional",
        ).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print(
        f"======================= Parameter Report of Diffusion Model ======================="
    )
    report_parameters(nn_diffusion)
    print(
        f"=============================================================================="
    )

    if args.diffusion == "ddpm":
        from policies.dp.diffuser.diffusion.ddpm import DDPM

        x_max = (
            torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
        )
        x_min = (
            torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
        )
        agent = DDPM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            diffusion_steps=args.sample_steps,
            x_max=x_max,
            x_min=x_min,
            optim_params={"lr": args.lr},
        )
    elif args.diffusion == "edm":
        from .diffuser.diffusion.edm import EDM

        agent = EDM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            optim_params={"lr": args.lr},
        )
    else:
        raise NotImplementedError
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    if args.mode == "train":
        # ----------------- Training ----------------------
        n_gradient_step = 0
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            # get condition
            nobs = batch["obs"]
            condition = {}
            for k in nobs.keys():
                condition[k] = nobs[k][:, : args.obs_steps, :].to(
                    args.device
                )  # Batch size, Obs_steps, Self.shape

            naction = batch["action"].to(
                args.device
            )  # Batch size, Sample sequence length, Action length

            # update diffusion
            diffusion_loss = agent.update(naction, condition)["loss"]
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    "step": n_gradient_step,
                    "total_time": time.time() - start_time,
                    "avg_diffusion_loss": np.mean(diffusion_loss_list),
                }
                logger.log(metrics, category="train")
                diffusion_loss_list = []

            if n_gradient_step % args.save_freq == 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)

            if n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                metrics = {"step": n_gradient_step}
                metrics.update(eval(args, env, dataset, agent, n_gradient_step))
                logger.log(metrics, category="eval")
                agent.model.train()
                agent.model_ema.train()

            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish(agent)
                break
    elif args.mode == "eval":
        # ----------------- eval ----------------------
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for eval")
        agent.model.eval()
        agent.model_ema.eval()

        metrics = {"step": 0}
        metrics.update(eval(args, env, dataset, agent, 0))
        logger.log(metrics, category="eval")

    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
