#!/usr/bin/env python3
"""
Standalone training script using pure MuJoCo (no DISCOVERSE dependency).

Usage:
    python scripts/train_standalone.py \
        --config configs/tasks/put_bottle_into_pot.yaml \
        --constraints_dir outputs/put_bottle_into_pot/constraints \
        --total_timesteps 100000
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Standalone PPO training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to task config YAML file")
    parser.add_argument("--constraints_dir", type=str, required=True,
                        help="Directory containing generated constraints")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs and models")
    parser.add_argument("--total_timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="PPO batch size")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per PPO update")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_freq", type=int, default=10000,
                        help="Evaluation frequency (timesteps)")
    parser.add_argument("--save_freq", type=int, default=50000,
                        help="Model save frequency (timesteps)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0, 1, or 2)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task_name = config.get("task_name", "unknown_task")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
            task_name,
            f"train_{timestamp}"
        )

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("STANDALONE PPO TRAINING")
    print("=" * 60)
    print(f"\nTask: {task_name}")
    print(f"Constraints: {args.constraints_dir}")
    print(f"Output: {output_dir}")
    print(f"Total timesteps: {args.total_timesteps}")

    # Set seed
    np.random.seed(args.seed)

    # Import SBX PPO
    try:
        from sbx import PPO
        print("\nUsing SBX PPO (JAX-based)")
    except ImportError:
        try:
            from stable_baselines3 import PPO
            print("\nUsing Stable Baselines3 PPO")
        except ImportError:
            print("Error: Please install sbx-rl or stable-baselines3")
            print("  pip install sbx-rl")
            print("  or")
            print("  pip install stable-baselines3")
            return 1

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor

    # Import our standalone environment
    from src.standalone_env import make_standalone_env

    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    constraints_dir = args.constraints_dir
    if not os.path.isabs(constraints_dir):
        constraints_dir = os.path.join(base_dir, constraints_dir)

    # Create environment factory
    def make_env():
        def _init():
            env = make_standalone_env(
                config_path=config_path,
                constraints_dir=constraints_dir,
                render_mode=None,
            )
            env = Monitor(env)
            return env
        return _init

    print("\nCreating training environment...")
    train_env = DummyVecEnv([make_env()])

    # Optionally normalize observations and rewards
    use_normalization = config.get("use_normalization", True)
    if use_normalization:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env()])
    if use_normalization:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize rewards for eval
            clip_obs=10.0,
            training=False,
        )

    # Custom callback for detailed logging
    class DetailedLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_stages = []
            self.episode_grasp_enclosure = []
            self.current_rewards = 0
            self.current_length = 0
            self.max_stage_reached = 1
            self.current_grasp_enclosure = 0.0

        def _on_step(self) -> bool:
            self.current_rewards += self.locals['rewards'][0]
            self.current_length += 1

            # Track stage progression
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'stage' in info:
                    self.max_stage_reached = max(self.max_stage_reached, info['stage'])
                if 'grasp_enclosure_reward' in info:
                    self.current_grasp_enclosure += info['grasp_enclosure_reward']

            if self.locals['dones'][0]:
                self.episode_rewards.append(self.current_rewards)
                self.episode_lengths.append(self.current_length)
                self.episode_stages.append(self.max_stage_reached)
                self.episode_grasp_enclosure.append(self.current_grasp_enclosure)

                self.current_rewards = 0
                self.current_length = 0
                self.max_stage_reached = 1
                self.current_grasp_enclosure = 0.0

                # Log every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    n = min(10, len(self.episode_rewards))
                    avg_reward = np.mean(self.episode_rewards[-n:])
                    avg_length = np.mean(self.episode_lengths[-n:])
                    avg_stage = np.mean(self.episode_stages[-n:])
                    avg_grasp_enclosure = np.mean(self.episode_grasp_enclosure[-n:])
                    print(f"  Episodes: {len(self.episode_rewards):4d} | "
                          f"Avg Reward: {avg_reward:8.2f} | "
                          f"Avg Length: {avg_length:6.0f} | "
                          f"Avg Stage: {avg_stage:.1f} | "
                          f"Avg Enclosure: {avg_grasp_enclosure:6.3f}")

            return True

        def _on_training_end(self):
            # Save episode statistics
            stats_path = os.path.join(output_dir, "episode_stats.npz")
            np.savez(
                stats_path,
                rewards=np.array(self.episode_rewards),
                lengths=np.array(self.episode_lengths),
                stages=np.array(self.episode_stages),
                grasp_enclosure=np.array(self.episode_grasp_enclosure),
            )
            print(f"\nSaved episode stats to {stats_path}")

    # Get PPO parameters from config
    ppo_config = config.get("ppo", {})
    n_steps = args.n_steps or ppo_config.get("n_steps", 2048)
    batch_size = args.batch_size or ppo_config.get("batch_size", 64)
    learning_rate = args.learning_rate or ppo_config.get("learning_rate", 3e-4)
    n_epochs = ppo_config.get("n_epochs", 10)
    gamma = ppo_config.get("gamma", 0.99)
    clip_range = ppo_config.get("clip_range", 0.2)
    ent_coef = ppo_config.get("ent_coef", 0.01)

    # Check if tensorboard is available
    try:
        import tensorboard
        tb_log_dir = log_dir
        print("\nTensorboard logging enabled")
    except ImportError:
        tb_log_dir = None
        print("\nTensorboard not installed, logging disabled")

    # Create PPO model
    print("\nCreating PPO model...")
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  gamma: {gamma}")

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        tensorboard_log=tb_log_dir,
        verbose=args.verbose,
        seed=args.seed,
    )

    # Setup callbacks
    callbacks = [
        DetailedLoggingCallback(verbose=args.verbose),
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=model_dir,
            name_prefix="ppo_checkpoint",
        ),
    ]

    # Save training config
    train_config = {
        "task_name": task_name,
        "constraints_dir": constraints_dir,
        "total_timesteps": args.total_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "seed": args.seed,
        "use_normalization": use_normalization,
    }
    with open(os.path.join(output_dir, "train_config.yaml"), "w") as f:
        yaml.dump(train_config, f)

    # Train
    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"{'=' * 60}\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")

    # Save VecNormalize stats if used
    if use_normalization:
        vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
        train_env.save(vec_norm_path)
        print(f"Saved normalization stats to {vec_norm_path}")

    # Print final statistics
    callback = callbacks[0]
    if callback.episode_rewards:
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total episodes: {len(callback.episode_rewards)}")
        print(f"Final avg reward (last 100): {np.mean(callback.episode_rewards[-100:]):.2f}")
        print(f"Final avg length (last 100): {np.mean(callback.episode_lengths[-100:]):.0f}")
        print(f"Final avg stage (last 100): {np.mean(callback.episode_stages[-100:]):.2f}")
        print(f"\nResults saved to: {output_dir}")

    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
