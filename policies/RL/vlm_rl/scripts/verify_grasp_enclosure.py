#!/usr/bin/env python3
"""
Verify grasp enclosure reward by random sampling arm configurations.

Searches for joint configurations that place the grasp keypoint between the two finger pads,
then reports the best enclosure reward and related metrics. Optional viewer to inspect.
"""

import argparse
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import mujoco
import mujoco.viewer
import yaml


def compute_enclosure_metrics(env, keypoints: np.ndarray) -> Dict[str, Any]:
    left_id = env._left_pad_id if env._left_pad_id is not None else -1
    right_id = env._right_pad_id if env._right_pad_id is not None else -1
    if left_id < 0 or right_id < 0:
        return {
            "valid": False,
            "gap": None,
            "axial": None,
            "radial": None,
            "open_fraction": None,
            "inside": False,
        }

    left_pos = env.mj_data.xpos[left_id].copy()
    right_pos = env.mj_data.xpos[right_id].copy()
    gap = float(np.linalg.norm(right_pos - left_pos))
    if gap < 1e-6:
        return {
            "valid": False,
            "gap": gap,
            "axial": None,
            "radial": None,
            "open_fraction": None,
            "inside": False,
        }

    axis = (right_pos - left_pos) / gap
    center = 0.5 * (left_pos + right_pos)
    point = keypoints[env.primary_grasp_kp_idx]

    rel = point - center
    axial = abs(float(np.dot(rel, axis)))
    radial = float(np.linalg.norm(rel - np.dot(rel, axis) * axis))

    half_gap = 0.5 * gap + env.grasp_enclosure_margin
    inside = (axial <= half_gap) and (radial <= env.grasp_enclosure_radius)

    open_frac = env._get_gripper_open_fraction()
    closed_enough = (open_frac is None) or (open_frac <= env.grasp_enclosure_max_open)
    if not closed_enough:
        inside = False

    return {
        "valid": True,
        "gap": gap,
        "axial": axial,
        "radial": radial,
        "open_fraction": open_frac,
        "inside": inside,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify grasp enclosure reward via random sampling")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "tasks", "put_bottle_into_pot.yaml"),
        help="Task config path (relative to vlm_rl or absolute)",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=os.path.join("outputs", "put_bottle_into_pot", "constraints"),
        help="Constraints directory (relative to vlm_rl or absolute)",
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-fraction", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show", action="store_true", help="Show best configuration in viewer")
    parser.add_argument("--sleep", type=float, default=0.01)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vlm_dir = os.path.abspath(os.path.join(script_dir, ".."))

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(vlm_dir, config_path)
    constraints_dir = args.constraints
    if not os.path.isabs(constraints_dir):
        constraints_dir = os.path.join(vlm_dir, constraints_dir)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sys.path.append(vlm_dir)
    from src.standalone_env import StandaloneVLMRLEnv

    mjcf_path = os.path.join(vlm_dir, config["mjcf_file_path"])
    env = StandaloneVLMRLEnv(
        mjcf_path=mjcf_path,
        constraints_dir=constraints_dir,
        config=config,
    )
    env.reset()

    print(f"Sampling: {args.samples} | top_k: {args.top_k} | close_fraction: {args.close_fraction}")

    rng = np.random.default_rng(args.seed)
    base_qpos = env.mj_data.qpos.copy()

    if env._gripper_qpos_addr is None or not env.gripper_ctrl_ranges:
        print("Gripper joint not found; enclosure reward may be unreliable.")

    top = []
    valid = 0
    skipped = 0

    for i in range(args.samples):
        env.mj_data.qpos[:] = base_qpos
        env.mj_data.qvel[:] = 0

        for jid, (low, high) in zip(env.arm_joint_ids, env.arm_joint_ranges):
            addr = env.mj_model.jnt_qposadr[jid]
            env.mj_data.qpos[addr] = rng.uniform(low, high)

        # Set gripper to a closed fraction
        if env._gripper_qpos_addr is not None and env.gripper_ctrl_ranges:
            low, high = env.gripper_ctrl_ranges[0]
            env.mj_data.qpos[env._gripper_qpos_addr] = low + args.close_fraction * (high - low)

        try:
            mujoco.mj_forward(env.mj_model, env.mj_data)
        except Exception as exc:
            if skipped < 3:
                print(f"[WARN] mj_forward failed at sample {i}: {exc}")
            skipped += 1
            continue

        valid += 1

        keypoints = env.keypoint_tracker.get_positions()
        reward = env._compute_grasp_enclosure_reward(keypoints)
        metrics = compute_enclosure_metrics(env, keypoints)

        if len(top) < args.top_k or reward > top[-1]["reward"]:
            entry = {
                "reward": reward,
                "metrics": metrics,
                "qpos": env.mj_data.qpos.copy(),
            }
            top.append(entry)
            top.sort(key=lambda x: x["reward"], reverse=True)
            top = top[:args.top_k]

    print(f"Valid samples: {valid} | Skipped: {skipped}")
    print("Top results:")
    for idx, item in enumerate(top, start=1):
        m = item["metrics"]
        print(
            f"{idx:2d}) reward={item['reward']:+.3f} | inside={m['inside']} | "
            f"gap={m['gap']:.4f} | axial={m['axial']:.4f} | radial={m['radial']:.4f} | "
            f"open_frac={m['open_fraction'] if m['open_fraction'] is not None else 'N/A'}"
        )

    if args.show and top:
        env.mj_data.qpos[:] = top[0]["qpos"]
        env.mj_data.qvel[:] = 0
        mujoco.mj_forward(env.mj_model, env.mj_data)

        with mujoco.viewer.launch_passive(env.mj_model, env.mj_data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(env.mj_model, env.mj_data)
                viewer.sync()
                time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
