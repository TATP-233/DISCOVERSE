#!/usr/bin/env python3
"""
Train a policy using VLM-generated constraints.

This script trains a PPO agent using rewards derived from VLM-generated
constraint functions.
"""

import argparse
import os
import sys
import yaml
from datetime import datetime

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Train policy with VLM constraints")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to task config YAML file")
    parser.add_argument("--constraints_dir", type=str, required=True,
                        help="Directory containing generated constraints")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs and models")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="PPO batch size")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per PPO update")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained model to continue training")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N updates")
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

    print(f"Task: {task_name}")
    print(f"Constraints: {args.constraints_dir}")
    print(f"Output: {output_dir}")

    # Import required libraries
    try:
        from sbx import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("Error: Please install sbx: pip install sbx-rl")
        return 1

    # Import DISCOVERSE components
    task_module = config.get("task_module", "discoverse.examples.tasks_mmk2.kiwi_pick")
    print(f"Loading task module: {task_module}")

    try:
        exec(f"from {task_module} import SimNode, cfg", globals())
    except ImportError as e:
        print(f"Error importing task module: {e}")
        return 1

    # Import our environment
    from src.env import VLMRLEnv

    # Configure task
    cfg_updates = config.get("cfg_updates", {})
    for key, value in cfg_updates.items():
        setattr(cfg, key, value)

    cfg.headless = not args.render
    cfg.sync = True

    # Set seed
    np.random.seed(args.seed)

    # Create environment factory
    def make_env():
        def _init():
            task_base = SimNode(cfg)
            env = VLMRLEnv(
                task_base=task_base,
                config=config,
                constraints_dir=args.constraints_dir,
                render_mode=args.render,
            )
            return env
        return _init

    print("Creating environment...")
    env = DummyVecEnv([make_env()])

    # Custom callback for logging
    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.current_rewards = 0
            self.current_length = 0

        def _on_step(self) -> bool:
            self.current_rewards += self.locals['rewards'][0]
            self.current_length += 1

            if self.locals['dones'][0]:
                self.episode_rewards.append(self.current_rewards)
                self.episode_lengths.append(self.current_length)
                self.current_rewards = 0
                self.current_length = 0

                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"Episodes: {len(self.episode_rewards)}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Avg Length: {avg_length:.0f}")

            return True

    # Create or load model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path, env=env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            learning_rate=args.learning_rate,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=log_dir,
            verbose=1,
            seed=args.seed,
        )

    # Save training config
    train_config = {
        "task_name": task_name,
        "constraints_dir": args.constraints_dir,
        "total_timesteps": args.total_timesteps,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    with open(os.path.join(output_dir, "train_config.yaml"), "w") as f:
        yaml.dump(train_config, f)

    # Train
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    callback = RewardCallback()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=args.log_interval,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")

    # Save training statistics
    stats = {
        "total_episodes": len(callback.episode_rewards),
        "final_avg_reward": float(np.mean(callback.episode_rewards[-100:])) if callback.episode_rewards else 0,
        "final_avg_length": float(np.mean(callback.episode_lengths[-100:])) if callback.episode_lengths else 0,
    }
    with open(os.path.join(output_dir, "training_stats.yaml"), "w") as f:
        yaml.dump(stats, f)

    print(f"\nTraining complete!")
    print(f"Final average reward (last 100 episodes): {stats['final_avg_reward']:.2f}")
    print(f"Results saved to: {output_dir}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
