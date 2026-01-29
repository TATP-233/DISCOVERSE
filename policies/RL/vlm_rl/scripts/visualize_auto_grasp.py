#!/usr/bin/env python3
"""
Visualize auto-grasp behavior in MuJoCo viewer.

Moves the jaw center toward the grasp keypoint, holds alignment to trigger auto-grasp,
then lifts the target point. Requires a GUI for mujoco.viewer.
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize auto-grasp with jaw-center IK")
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
    parser.add_argument("--approach-steps", type=int, default=300)
    parser.add_argument("--pregrasp-dz", type=float, default=0.08)
    parser.add_argument("--descend-steps", type=int, default=150)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--close-steps", type=int, default=120)
    parser.add_argument("--lift-steps", type=int, default=200)
    parser.add_argument("--lift-dz", type=float, default=0.15)
    parser.add_argument("--open-steps", type=int, default=30)
    parser.add_argument("--action-scale", type=float, default=0.6)
    parser.add_argument("--rot-weight", type=float, default=0.5)
    parser.add_argument("--axis-mode", choices=["pca", "pair"], default="pca")
    parser.add_argument("--axis-pair", type=str, default="0,4",
                        help="Keypoint pair for axis mode=pair, format: i,j")
    parser.add_argument("--no-freeze-target", action="store_true",
                        help="Track moving keypoint instead of freezing initial target")
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

    config = load_config(config_path)

    sys.path.append(vlm_dir)
    from src.standalone_env import StandaloneVLMRLEnv

    mjcf_path = os.path.join(vlm_dir, config["mjcf_file_path"])
    env = StandaloneVLMRLEnv(
        mjcf_path=mjcf_path,
        constraints_dir=constraints_dir,
        config=config,
    )
    env.reset()

    left_id = env._left_pad_id if env._left_pad_id is not None and env._left_pad_id >= 0 else None
    right_id = env._right_pad_id if env._right_pad_id is not None and env._right_pad_id >= 0 else None
    use_jaw_center = left_id is not None and right_id is not None

    site_name = config.get("end_effector_site", "gripper")
    site_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        site_id = None

    arm_dof_ids = np.array([env.mj_model.jnt_dofadr[jid] for jid in env.arm_joint_ids], dtype=int)
    max_delta = float(config.get("max_action_delta", 0.05))
    action_scale = float(args.action_scale)
    rot_weight = float(args.rot_weight)
    freeze_target = not args.no_freeze_target

    try:
        axis_pair = tuple(int(x) for x in args.axis_pair.split(","))
    except ValueError:
        axis_pair = (0, 4)

    def get_center() -> np.ndarray:
        if use_jaw_center:
            left_pos = env.mj_data.xpos[left_id].copy()
            right_pos = env.mj_data.xpos[right_id].copy()
            return 0.5 * (left_pos + right_pos)
        return env._get_end_effector_pos()

    def get_jacobian() -> np.ndarray:
        jac = np.zeros((3, env.mj_model.nv))
        if use_jaw_center:
            jac_l = np.zeros((3, env.mj_model.nv))
            jac_r = np.zeros((3, env.mj_model.nv))
            mujoco.mj_jacBody(env.mj_model, env.mj_data, jac_l, None, left_id)
            mujoco.mj_jacBody(env.mj_model, env.mj_data, jac_r, None, right_id)
            jac = 0.5 * (jac_l + jac_r)
        elif site_id is not None:
            mujoco.mj_jacSite(env.mj_model, env.mj_data, jac, None, site_id)
        return jac[:, arm_dof_ids]

    def compute_object_axis(keypoints: np.ndarray) -> Optional[np.ndarray]:
        ranges = getattr(env, "object_keypoint_ranges", {}) or {}
        obj_indices = None
        for _, (start, end) in ranges.items():
            if start <= kp_idx < end:
                obj_indices = list(range(start, end))
                break
        if not obj_indices:
            return None
        points = keypoints[obj_indices]
        if len(points) < 2:
            return None
        if args.axis_mode == "pair":
            i, j = axis_pair
            if i < 0 or j < 0 or i >= len(points) or j >= len(points):
                return None
            axis = points[j] - points[i]
        else:
            mean = np.mean(points, axis=0)
            centered = points - mean
            cov = centered.T @ centered
            vals, vecs = np.linalg.eigh(cov)
            axis = vecs[:, int(np.argmax(vals))]
        if np.linalg.norm(axis) < 1e-6:
            return None
        return axis / (np.linalg.norm(axis) + 1e-8)

    def get_jaw_axis() -> Optional[np.ndarray]:
        if left_id is None or right_id is None:
            return None
        left_pos = env.mj_data.xpos[left_id].copy()
        right_pos = env.mj_data.xpos[right_id].copy()
        axis = right_pos - left_pos
        if np.linalg.norm(axis) < 1e-6:
            return None
        return axis / (np.linalg.norm(axis) + 1e-8)

    def step_to(target: np.ndarray) -> None:
        center = get_center()
        err = target - center
        Jp = get_jacobian()

        # Orientation alignment: keep jaw axis perpendicular to object axis
        rot_err = np.zeros(3, dtype=np.float32)
        if rot_weight > 0.0 and site_id is not None:
            keypoints = env.keypoint_tracker.get_positions()
            obj_axis = compute_object_axis(keypoints)
            jaw_axis = get_jaw_axis()
            if obj_axis is not None and jaw_axis is not None:
                # Desired jaw axis: current axis projected onto plane perpendicular to object axis
                desired = jaw_axis - np.dot(jaw_axis, obj_axis) * obj_axis
                if np.linalg.norm(desired) < 1e-6:
                    # Fallback: pick any perpendicular direction
                    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    if abs(np.dot(tmp, obj_axis)) > 0.9:
                        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    desired = np.cross(obj_axis, tmp)
                desired = desired / (np.linalg.norm(desired) + 1e-8)
                rot_err = np.cross(jaw_axis, desired)

        Jr = np.zeros((3, env.mj_model.nv))
        if site_id is not None and rot_weight > 0.0:
            mujoco.mj_jacSite(env.mj_model, env.mj_data, None, Jr, site_id)
            Jr = Jr[:, arm_dof_ids]

        if rot_weight > 0.0:
            J = np.vstack([Jp, rot_weight * Jr])
            e = np.concatenate([err, rot_weight * rot_err])
        else:
            J = Jp
            e = err

        lam = 0.1
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(JJt.shape[0]), e)
        arm_action = np.clip(dq / max_delta, -1.0, 1.0) * action_scale
        if env.gripper_action_dim:
            action = np.concatenate([arm_action, np.zeros(1, dtype=np.float32)])
        else:
            action = arm_action
        env.step(action)

    kp_idx = env.primary_grasp_kp_idx

    with mujoco.viewer.launch_passive(env.mj_model, env.mj_data) as viewer:
        # Open gripper before approach
        for _ in range(args.open_steps):
            if not viewer.is_running():
                return 0
            if env.gripper_action_dim:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                action[-1] = 1.0
                env.step(action)
            else:
                env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
            viewer.sync()
            time.sleep(args.sleep)

        # Cache initial target to avoid chasing the bottle if it moves
        kp_initial = env.keypoint_tracker.get_position(kp_idx)

        # Approach
        for _ in range(args.approach_steps):
            if not viewer.is_running():
                return 0
            kp_target = env.keypoint_tracker.get_position(kp_idx) if not freeze_target else kp_initial
            pregrasp = kp_target.copy()
            pregrasp[2] += args.pregrasp_dz
            step_to(pregrasp)
            viewer.sync()
            time.sleep(args.sleep)

        # Descend straight down in Z while holding XY
        for i in range(args.descend_steps):
            if not viewer.is_running():
                return 0
            kp_target = env.keypoint_tracker.get_position(kp_idx) if not freeze_target else kp_initial
            z = (kp_target[2] + args.pregrasp_dz) + (kp_target[2] - (kp_target[2] + args.pregrasp_dz)) * ((i + 1) / max(args.descend_steps, 1))
            descend_target = np.array([kp_target[0], kp_target[1], z], dtype=np.float32)
            step_to(descend_target)
            viewer.sync()
            time.sleep(args.sleep)

        # Hold alignment
        for _ in range(args.hold_steps):
            if not viewer.is_running():
                return 0
            kp_target = env.keypoint_tracker.get_position(kp_idx) if not freeze_target else kp_initial
            step_to(kp_target)
            viewer.sync()
            time.sleep(args.sleep)

        # Let auto-grasp close
        for _ in range(args.close_steps):
            if not viewer.is_running():
                return 0
            if env.gripper_action_dim:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            env.step(action)
            viewer.sync()
            time.sleep(args.sleep)

        # Lift
        kp_now = env.keypoint_tracker.get_position(kp_idx)
        lift_target = kp_now.copy()
        lift_target[2] += args.lift_dz
        for _ in range(args.lift_steps):
            if not viewer.is_running():
                return 0
            step_to(lift_target)
            viewer.sync()
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
