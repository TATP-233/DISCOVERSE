#!/usr/bin/env python3
"""
Run inference with a trained policy.

This script evaluates a trained model on the task and reports performance.
"""

import argparse
import os
import sys
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to task config YAML file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--constraints_dir", type=str, required=True,
                        help="Directory containing constraints")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save episode videos")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task_name = config.get("task_name", "unknown_task")

    print(f"Task: {task_name}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")

    # Import required libraries
    try:
        from sbx import PPO
    except ImportError:
        print("Error: Please install sbx: pip install sbx-rl")
        return 1

    # Import DISCOVERSE components
    task_module = config.get("task_module", "discoverse.examples.tasks_mmk2.kiwi_pick")

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

    # Create environment
    print("Creating environment...")
    task_base = SimNode(cfg)
    env = VLMRLEnv(
        task_base=task_base,
        config=config,
        constraints_dir=args.constraints_dir,
        render_mode=args.render,
    )

    # Load model
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    # Run evaluation
    print(f"\nRunning {args.episodes} evaluation episodes...")

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    stage_reached = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_stage = 1

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Track max stage reached
            if "stage" in info:
                max_stage = max(max_stage, info["stage"])

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(terminated and not truncated)
        stage_reached.append(max_stage)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{args.episodes}: "
                  f"Reward={total_reward:.2f}, Steps={steps}, "
                  f"Success={terminated and not truncated}, Stage={max_stage}")

    # Calculate statistics
    results = {
        "num_episodes": args.episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(episode_successes)),
        "mean_stage_reached": float(np.mean(stage_reached)),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:        {results['num_episodes']}")
    print(f"Mean Reward:     {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean Length:     {results['mean_length']:.0f} +/- {results['std_length']:.0f}")
    print(f"Success Rate:    {results['success_rate']*100:.1f}%")
    print(f"Mean Stage:      {results['mean_stage_reached']:.2f}")
    print("=" * 50)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "evaluation_results.yaml")
        with open(results_path, "w") as f:
            yaml.dump(results, f)
        print(f"\nResults saved to: {results_path}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
