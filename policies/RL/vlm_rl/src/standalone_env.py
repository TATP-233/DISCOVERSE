"""
StandaloneVLMRLEnv: Gymnasium environment for VLM-guided RL without DISCOVERSE dependency.

Uses pure MuJoCo for simulation, integrating keypoint tracking and VLM-generated constraints.
"""

import numpy as np
import mujoco
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any
import os
import json
import yaml

from .keypoint_tracker import KeypointTracker
from .constraint_generator import ConstraintLoader, GenerationResult
from .reward_adapter import ConstraintRewardAdapter


class StandaloneVLMRLEnv(gymnasium.Env):
    """
    Gymnasium environment using pure MuJoCo with VLM-generated constraint rewards.

    Does not require DISCOVERSE - works with any MuJoCo MJCF file.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mjcf_path: str,
        constraints_dir: str,
        config: Dict[str, Any],
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            mjcf_path: Path to MuJoCo MJCF XML file
            constraints_dir: Directory containing generated constraints
            config: Configuration dictionary
            render_mode: "human" for window, "rgb_array" for images, None for no render
        """
        super().__init__()

        self.mjcf_path = mjcf_path
        self.constraints_dir = constraints_dir
        self.config = config
        self.render_mode = render_mode

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Configuration
        self.end_effector_body = config.get("end_effector_body", "robotiq_base")
        self.max_steps = config.get("max_steps", 500)
        self.max_time = config.get("max_time", 15.0)
        self.dt = self.mj_model.opt.timestep
        self.frame_skip = config.get("frame_skip", 5)  # Steps per action

        # Arm joint configuration - find arm joints
        self._setup_arm_joints()
        # Gripper actuator configuration
        self._setup_gripper_actuators()

        # Gripper control
        self.enable_gripper_control = self.config.get("enable_gripper_control", False)
        self.gripper_action_mode = self.config.get("gripper_action_mode", "absolute")
        self.gripper_action_dim = 1 if (self.enable_gripper_control and self.gripper_actuator_ids) else 0

        # Action space: arm joint position control
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_arm_joints + self.gripper_action_dim,),
            dtype=np.float32
        )

        # Load constraints and keypoints
        self._load_constraints()

        # Initialize keypoint tracker
        self._init_keypoint_tracker()

        # Grasp metadata (from generated constraints)
        self._init_grasp_metadata()

        # Observation space: qpos + qvel + ee_pos + keypoints
        obs_dim = (
            self.mj_model.nq +  # Joint positions
            self.mj_model.nv +  # Joint velocities
            3 +  # End effector position
            self.num_keypoints * 3  # Keypoint positions
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State
        self.current_step = 0
        self._initial_qpos = self.mj_data.qpos.copy()
        self._initial_qvel = self.mj_data.qvel.copy()

        # Auto-grasp state
        self._reset_grasp_state()

        # Store home position
        self._store_home_position()

        # Renderer (for rgb_array mode)
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.mj_model, 480, 640)

    def _setup_arm_joints(self):
        """Find and setup arm joint indices."""
        # Find arm joints - specifically joints 1-7 for Franka Panda
        self.arm_joint_ids = []
        self.arm_joint_names = []

        # Exclude patterns for gripper and other non-arm joints
        exclude_patterns = [
            'gripper', 'finger', 'driver', 'spring', 'follower',
            'left_', 'right_', 'bottle', 'pot', 'duster', 'towel'
        ]

        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                continue

            joint_name_lower = joint_name.lower()

            # Skip if matches any exclude pattern
            if any(pat in joint_name_lower for pat in exclude_patterns):
                continue

            # Include joints named 'joint1' through 'joint7' or containing 'arm'
            is_arm_joint = False
            if 'arm' in joint_name_lower:
                is_arm_joint = True
            elif joint_name_lower.startswith('joint') and len(joint_name_lower) <= 7:
                # Check if it's joint1-joint7
                try:
                    num = int(joint_name_lower.replace('joint', ''))
                    if 1 <= num <= 7:
                        is_arm_joint = True
                except ValueError:
                    pass

            if is_arm_joint:
                joint_type = self.mj_model.jnt_type[i]
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    self.arm_joint_ids.append(i)
                    self.arm_joint_names.append(joint_name)

        # Also find gripper joints for control
        self.gripper_joint_ids = []
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and ('driver' in joint_name.lower() or 'finger' in joint_name.lower()):
                joint_type = self.mj_model.jnt_type[i]
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    self.gripper_joint_ids.append(i)

        self.num_arm_joints = len(self.arm_joint_ids)

        # Get joint limits
        self.arm_joint_ranges = []
        for jid in self.arm_joint_ids:
            qpos_addr = self.mj_model.jnt_qposadr[jid]
            if self.mj_model.jnt_limited[jid]:
                low = self.mj_model.jnt_range[jid, 0]
                high = self.mj_model.jnt_range[jid, 1]
            else:
                low, high = -np.pi, np.pi
            self.arm_joint_ranges.append((low, high))

        print(f"Found {self.num_arm_joints} arm joints: {self.arm_joint_names}")

    def _setup_gripper_actuators(self) -> None:
        """Find gripper actuators and control ranges."""
        self.gripper_actuator_ids = []
        self.gripper_ctrl_ranges = []
        self._gripper_joint_id = None
        self._gripper_qpos_addr = None
        self._left_pad_id = None
        self._right_pad_id = None

        for act_id in range(self.mj_model.nu):
            act_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            name_lower = act_name.lower() if act_name else ""

            joint_id = int(self.mj_model.actuator_trnid[act_id, 0])
            is_gripper = False
            if joint_id in getattr(self, "gripper_joint_ids", []):
                is_gripper = True
            elif any(token in name_lower for token in ["gripper", "finger", "driver"]):
                is_gripper = True

            if not is_gripper:
                continue

            # Determine control range
            if self.mj_model.actuator_ctrllimited[act_id]:
                low, high = self.mj_model.actuator_ctrlrange[act_id]
            else:
                low, high = 0.0, 1.0
                if joint_id >= 0 and self.mj_model.jnt_limited[joint_id]:
                    low, high = self.mj_model.jnt_range[joint_id]

            self.gripper_actuator_ids.append(act_id)
            self.gripper_ctrl_ranges.append((float(low), float(high)))

            if self._gripper_joint_id is None and joint_id >= 0:
                self._gripper_joint_id = joint_id
                self._gripper_qpos_addr = int(self.mj_model.jnt_qposadr[joint_id])

        if self.gripper_actuator_ids:
            print(f"Found {len(self.gripper_actuator_ids)} gripper actuators: {self.gripper_actuator_ids}")

        # Cache finger pad geom ids for accurate jaw center/axis (geom positions are more accurate than body positions)
        self._left_pad_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
        self._right_pad_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")

        # Fallback to body ids if geom not found
        self._left_pad_id = None
        self._right_pad_id = None
        if self._left_pad_geom_id < 0 or self._right_pad_geom_id < 0:
            self._left_pad_geom_id = None
            self._right_pad_geom_id = None
            self._left_pad_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_pad")
            self._right_pad_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
            if self._left_pad_id < 0 or self._right_pad_id < 0:
                self._left_pad_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_follower")
                self._right_pad_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_follower")

    def _init_grasp_metadata(self) -> None:
        """Initialize grasp/release metadata from generation results."""
        self.primary_grasp_kp_idx = -1
        for kp_idx in self.generation_result.grasp_keypoints:
            if kp_idx is not None and kp_idx >= 0:
                self.primary_grasp_kp_idx = int(kp_idx)
                break

        self.primary_release_stage = -1
        if self.primary_grasp_kp_idx >= 0:
            for stage_idx, kp_idx in enumerate(self.generation_result.release_keypoints, start=1):
                if kp_idx == self.primary_grasp_kp_idx:
                    self.primary_release_stage = stage_idx
                    break

        # Auto-grasp configuration
        self.auto_grasp = self.config.get("auto_grasp", False)
        self.auto_grasp_distance = float(self.config.get("auto_grasp_distance", 0.04))
        self.auto_grasp_align_steps = int(self.config.get("auto_grasp_align_steps", 5))
        self.auto_grasp_align_tol = float(self.config.get("auto_grasp_align_tol", 0.002))
        self.auto_grasp_orientation = bool(self.config.get("auto_grasp_orientation", False))
        self.auto_grasp_axis_mode = self.config.get("auto_grasp_axis_mode", "pca")
        self.auto_grasp_axis_pair = self.config.get("auto_grasp_axis_pair", [0, 4])
        self.auto_grasp_axis_perpendicular = bool(
            self.config.get("auto_grasp_axis_perpendicular", True)
        )
        self.auto_grasp_axis_cos_threshold = float(
            self.config.get("auto_grasp_axis_cos_threshold", 0.3)
        )
        self.auto_grasp_close_fraction = float(self.config.get("auto_grasp_close_fraction", 0.1))
        self.auto_grasp_close_speed = float(self.config.get("auto_grasp_close_speed", 0.02))
        self.auto_grasp_stuck_tol = float(self.config.get("auto_grasp_stuck_tol", 5e-4))
        self.auto_grasp_stuck_steps = int(self.config.get("auto_grasp_stuck_steps", 5))
        self.auto_grasp_follow_ratio = float(self.config.get("auto_grasp_follow_ratio", 0.6))
        self.auto_grasp_follow_cos = float(self.config.get("auto_grasp_follow_cos", 0.8))
        self.auto_grasp_follow_min_gripper_move = float(
            self.config.get("auto_grasp_follow_min_gripper_move", 1e-3)
        )
        self.auto_grasp_follow_steps = int(self.config.get("auto_grasp_follow_steps", 3))
        self.auto_grasp_disable_on_release_stage = bool(
            self.config.get("auto_grasp_disable_on_release_stage", True)
        )

        # Grasp enclosure reward (stage 1)
        self.grasp_enclosure_enable = bool(self.config.get("grasp_enclosure_enable", True))
        self.grasp_enclosure_weight = float(self.config.get("grasp_enclosure_weight", 0.5))
        self.grasp_enclosure_radius = float(self.config.get("grasp_enclosure_radius", 0.02))
        self.grasp_enclosure_margin = float(self.config.get("grasp_enclosure_margin", 0.005))
        self.grasp_enclosure_max_open = float(self.config.get("grasp_enclosure_max_open", 0.4))
        self.grasp_enclosure_penalty = bool(self.config.get("grasp_enclosure_penalty", True))

        # Require grasp confirmation for Stage 1 completion
        self.require_grasp_for_stage1 = bool(self.config.get("require_grasp_for_stage1", False))

    def _store_home_position(self):
        """Store the home position from keyconfig if available."""
        # Try to find home keyframe
        for i in range(self.mj_model.nkey):
            key_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name and 'home' in key_name.lower():
                mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, i)
                self._home_qpos = self.mj_data.qpos.copy()
                print(f"Found home keyframe: {key_name}")
                return

        # No home keyframe, use current position
        self._home_qpos = self.mj_data.qpos.copy()

    def _load_constraints(self):
        """Load constraints from directory."""
        result_path = os.path.join(self.constraints_dir, "generation_result.json")

        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Constraints not found: {result_path}")

        with open(result_path, "r") as f:
            data = json.load(f)

        self.generation_result = GenerationResult(
            num_stages=data["num_stages"],
            grasp_keypoints=data["grasp_keypoints"],
            release_keypoints=data["release_keypoints"],
            subgoal_constraints={int(k): v for k, v in data["subgoal_constraints"].items()},
            path_constraints={int(k): v for k, v in data["path_constraints"].items()},
            raw_output=data.get("raw_output", ""),
            output_dir=self.constraints_dir,
        )

        # Load constraint functions
        loader = ConstraintLoader()
        constraints = loader.load_from_result(self.generation_result)

        # Initialize reward adapter
        reward_config = self.config.get("reward_config", {})
        self.reward_adapter = ConstraintRewardAdapter(
            constraints=constraints,
            num_stages=self.generation_result.num_stages,
            grasp_keypoints=self.generation_result.grasp_keypoints,
            release_keypoints=self.generation_result.release_keypoints,
            **reward_config
        )

        print(f"Loaded {self.generation_result.num_stages} stages")

    def _init_keypoint_tracker(self):
        """Initialize keypoint tracker from saved keypoints."""
        keypoints_path = os.path.join(self.constraints_dir, "keypoints.yaml")

        if not os.path.exists(keypoints_path):
            raise FileNotFoundError(f"Keypoints not found: {keypoints_path}")

        # Use FullLoader to support Python tuples in YAML
        with open(keypoints_path, "r") as f:
            kp_data = yaml.load(f, Loader=yaml.FullLoader)

        self.keypoints_3d = np.array(kp_data["keypoints_3d"])
        self.object_keypoint_ranges = kp_data.get("object_keypoint_ranges", {})
        self.num_keypoints = len(self.keypoints_3d)

        # Initialize tracker
        self.keypoint_tracker = KeypointTracker(self.mj_model, self.mj_data)
        self.keypoint_tracker.register_keypoints(self.keypoints_3d)

        print(f"Loaded {self.num_keypoints} keypoints")

    def _get_end_effector_pos(self) -> np.ndarray:
        """Get end effector position.

        Prefers site over body for more accurate gripper tip position.
        """
        ee_mode = self.config.get("end_effector_mode", "site")

        # First try to use site (more accurate for gripper tip)
        site_name = self.config.get("end_effector_site", "gripper")
        site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id >= 0:
            ee_pos = self.mj_data.site_xpos[site_id].copy()
            return self._get_gripper_center_pos(ee_pos) if ee_mode == "jaw_center" else ee_pos

        # Fallback to body
        body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.end_effector_body
        )
        if body_id >= 0:
            ee_pos = self.mj_data.xpos[body_id].copy()
            return self._get_gripper_center_pos(ee_pos) if ee_mode == "jaw_center" else ee_pos

        print(f"Warning: Neither site '{site_name}' nor body '{self.end_effector_body}' found")
        return np.zeros(3)

    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        qpos = self.mj_data.qpos.copy()
        qvel = self.mj_data.qvel.copy()
        ee_pos = self._get_end_effector_pos()
        keypoints = self.keypoint_tracker.get_positions().flatten()

        return np.concatenate([qpos, qvel, ee_pos, keypoints]).astype(np.float32)

    def _apply_action(self, action: np.ndarray):
        """Apply action to arm joints using incremental control.

        Actions are treated as velocity/delta commands rather than absolute positions.
        This prevents sudden large joint movements that destabilize the simulation.
        """
        # Max joint position change per step (radians)
        max_delta = self.config.get("max_action_delta", 0.05)  # ~3 degrees per step

        arm_action = action[:self.num_arm_joints]
        gripper_action = None
        if self.gripper_action_dim:
            gripper_action = float(action[self.num_arm_joints])

        for i, (jid, (low, high)) in enumerate(zip(self.arm_joint_ids, self.arm_joint_ranges)):
            qpos_addr = self.mj_model.jnt_qposadr[jid]
            current_pos = self.mj_data.qpos[qpos_addr]

            # Incremental control: action in [-1, 1] maps to [-max_delta, max_delta]
            delta = arm_action[i] * max_delta
            target = np.clip(current_pos + delta, low, high)

            # Find actuator for this joint
            for act_id in range(self.mj_model.nu):
                if self.mj_model.actuator_trnid[act_id, 0] == jid:
                    self.mj_data.ctrl[act_id] = target
                    break

        if self._gripper_override_fraction is not None:
            self._apply_gripper_fraction(self._gripper_override_fraction)
        elif gripper_action is not None:
            self._apply_gripper_action(gripper_action)

    def _apply_gripper_action(self, action_value: float) -> None:
        """Apply gripper action to gripper actuators."""
        if not self.gripper_actuator_ids:
            return

        for act_id, (low, high) in zip(self.gripper_actuator_ids, self.gripper_ctrl_ranges):
            if self.gripper_action_mode == "binary":
                target = high if action_value > 0 else low
            else:
                # Absolute: map [-1, 1] -> [low, high]
                target = low + (action_value + 1.0) * 0.5 * (high - low)
            self.mj_data.ctrl[act_id] = float(np.clip(target, low, high))

    def _apply_gripper_fraction(self, open_fraction: float) -> None:
        """Apply gripper command using open fraction [0, 1]."""
        if not self.gripper_actuator_ids:
            return
        open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
        for act_id, (low, high) in zip(self.gripper_actuator_ids, self.gripper_ctrl_ranges):
            target = low + open_fraction * (high - low)
            self.mj_data.ctrl[act_id] = float(np.clip(target, low, high))

    def _get_gripper_qpos(self) -> Optional[float]:
        """Get gripper joint position (primary driver joint)."""
        if self._gripper_qpos_addr is None:
            return None
        return float(self.mj_data.qpos[self._gripper_qpos_addr])

    def _get_gripper_open_fraction(self) -> Optional[float]:
        """Estimate gripper open fraction using primary actuator range."""
        if not self.gripper_ctrl_ranges:
            return None
        qpos = self._get_gripper_qpos()
        if qpos is None:
            return None
        low, high = self.gripper_ctrl_ranges[0]
        if high <= low:
            return None
        return float(np.clip((qpos - low) / (high - low), 0.0, 1.0))

    def _reset_grasp_state(self) -> None:
        """Reset auto-grasp tracking state."""
        self._auto_grasp_active = False
        self._grasp_confirmed = False
        self._gripper_target_fraction = None
        self._gripper_hold_fraction = None
        self._gripper_override_fraction = None
        self._stuck_count = 0
        self._align_count = 0
        self._follow_count = 0
        self._last_grasp_distance = None
        self._prev_ee_pos = None
        self._prev_kp_pos = None
        self._prev_gripper_qpos = None
        self._prev_gripper_center = None
        self._grasp_object_keypoint_indices = None

    def _compute_auto_grasp_override(self, ee_pos: np.ndarray, keypoints: np.ndarray) -> Optional[float]:
        """Compute gripper override based on auto-grasp logic."""
        if not self.auto_grasp or not self.gripper_action_dim:
            return None

        if self._grasp_confirmed:
            return self._gripper_hold_fraction if self._gripper_hold_fraction is not None else self._gripper_target_fraction

        if self.auto_grasp_disable_on_release_stage and self.primary_release_stage > 0:
            if self.reward_adapter.current_stage >= self.primary_release_stage:
                self._auto_grasp_active = False
                return None

        if self.primary_grasp_kp_idx < 0 or self.primary_grasp_kp_idx >= len(keypoints):
            return None

        grasp_center = self._get_gripper_center_pos(ee_pos)
        distance = float(np.linalg.norm(grasp_center - keypoints[self.primary_grasp_kp_idx]))
        in_range = distance <= self.auto_grasp_distance
        if not in_range:
            self._auto_grasp_active = False
            self._align_count = 0
            self._last_grasp_distance = distance
            return None

        if self.auto_grasp_orientation:
            if not self._check_grasp_orientation(keypoints):
                self._auto_grasp_active = False
                self._align_count = 0
                self._last_grasp_distance = distance
                return None

        if self._last_grasp_distance is None:
            self._align_count = 1
        elif abs(distance - self._last_grasp_distance) <= self.auto_grasp_align_tol:
            self._align_count += 1
        else:
            self._align_count = 0

        if self._align_count >= self.auto_grasp_align_steps:
            self._auto_grasp_active = True

        self._last_grasp_distance = distance

        if not self._auto_grasp_active:
            return None

        current_fraction = self._get_gripper_open_fraction()
        if current_fraction is None:
            current_fraction = 1.0

        if self._gripper_target_fraction is None:
            self._gripper_target_fraction = current_fraction

        if self._grasp_confirmed:
            return self._gripper_hold_fraction if self._gripper_hold_fraction is not None else self._gripper_target_fraction

        if self._stuck_count >= self.auto_grasp_stuck_steps:
            self._gripper_target_fraction = current_fraction
            return self._gripper_target_fraction

        target_fraction = float(np.clip(self.auto_grasp_close_fraction, 0.0, 1.0))
        speed = float(max(self.auto_grasp_close_speed, 0.0))

        if self._gripper_target_fraction > target_fraction:
            self._gripper_target_fraction = max(self._gripper_target_fraction - speed, target_fraction)
        else:
            self._gripper_target_fraction = target_fraction

        return self._gripper_target_fraction

    def _check_grasp_orientation(self, keypoints: np.ndarray) -> bool:
        """Check if gripper is oriented properly relative to grasp object axis."""
        if self.primary_grasp_kp_idx < 0 or self.object_keypoint_ranges is None:
            return True

        if self._grasp_object_keypoint_indices is None:
            for obj_name, (start, end) in self.object_keypoint_ranges.items():
                if start <= self.primary_grasp_kp_idx < end:
                    self._grasp_object_keypoint_indices = list(range(start, end))
                    break

        if not self._grasp_object_keypoint_indices:
            return True

        obj_points = keypoints[self._grasp_object_keypoint_indices]
        if len(obj_points) < 2:
            return True

        axis = self._compute_object_axis(obj_points)
        if axis is None:
            return True

        jaw_axis = self._get_gripper_jaw_axis()
        if jaw_axis is None:
            return True

        axis = axis / (np.linalg.norm(axis) + 1e-8)
        jaw_axis = jaw_axis / (np.linalg.norm(jaw_axis) + 1e-8)
        alignment = abs(float(np.dot(axis, jaw_axis)))

        if self.auto_grasp_axis_perpendicular:
            return alignment <= self.auto_grasp_axis_cos_threshold
        return alignment >= self.auto_grasp_axis_cos_threshold

    def _compute_object_axis(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Compute principal axis of an object from keypoints."""
        if self.auto_grasp_axis_mode == "pair":
            try:
                idx_a, idx_b = self.auto_grasp_axis_pair
            except (TypeError, ValueError):
                return None
            if idx_a < 0 or idx_b < 0 or idx_a >= len(points) or idx_b >= len(points):
                return None
            axis = points[idx_b] - points[idx_a]
            if np.linalg.norm(axis) < 1e-6:
                return None
            return axis

        # PCA mode
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = centered.T @ centered
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        if np.linalg.norm(axis) < 1e-6:
            return None
        return axis

    def _get_finger_pad_positions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get left and right finger pad positions using geom (preferred) or body."""
        # Prefer geom positions (more accurate)
        if self._left_pad_geom_id is not None and self._right_pad_geom_id is not None:
            left_pos = self.mj_data.geom_xpos[self._left_pad_geom_id].copy()
            right_pos = self.mj_data.geom_xpos[self._right_pad_geom_id].copy()
            return left_pos, right_pos

        # Fallback to body positions
        left_id = self._left_pad_id if self._left_pad_id is not None else -1
        right_id = self._right_pad_id if self._right_pad_id is not None else -1
        if left_id < 0 or right_id < 0:
            return None, None

        left_pos = self.mj_data.xpos[left_id].copy()
        right_pos = self.mj_data.xpos[right_id].copy()
        return left_pos, right_pos

    def _get_gripper_jaw_axis(self) -> Optional[np.ndarray]:
        """Estimate jaw closing axis using left/right finger pads."""
        left_pos, right_pos = self._get_finger_pad_positions()
        if left_pos is None or right_pos is None:
            return None

        axis = right_pos - left_pos
        if np.linalg.norm(axis) < 1e-6:
            return None
        return axis

    def _get_gripper_center_pos(self, fallback: np.ndarray) -> np.ndarray:
        """Get gripper jaw center using finger pads, fallback to ee position."""
        left_pos, right_pos = self._get_finger_pad_positions()
        if left_pos is None or right_pos is None:
            return fallback
        return 0.5 * (left_pos + right_pos)

    def _compute_grasp_enclosure_reward(self, keypoints: np.ndarray) -> float:
        """Reward for having the grasp keypoint between the two finger pads.

        Uses geom positions (more accurate than body positions).
        """
        if not self.grasp_enclosure_enable:
            return 0.0
        if self.reward_adapter.current_stage != 1:
            return 0.0
        if self.primary_grasp_kp_idx < 0 or self.primary_grasp_kp_idx >= len(keypoints):
            return 0.0

        left_pos, right_pos = self._get_finger_pad_positions()
        if left_pos is None or right_pos is None:
            return 0.0

        gap = float(np.linalg.norm(right_pos - left_pos))
        if gap < 1e-6:
            return 0.0

        axis = (right_pos - left_pos) / gap
        center = 0.5 * (left_pos + right_pos)
        point = keypoints[self.primary_grasp_kp_idx]

        rel = point - center
        axial = abs(float(np.dot(rel, axis)))
        radial = float(np.linalg.norm(rel - np.dot(rel, axis) * axis))

        half_gap = 0.5 * gap + self.grasp_enclosure_margin
        inside = (axial <= half_gap) and (radial <= self.grasp_enclosure_radius)

        open_frac = self._get_gripper_open_fraction()
        closed_enough = (open_frac is None) or (open_frac <= self.grasp_enclosure_max_open)
        if not closed_enough:
            inside = False

        if inside:
            return self.grasp_enclosure_weight

        if not self.grasp_enclosure_penalty:
            return 0.0

        axial_excess = max(0.0, axial - half_gap)
        radial_excess = max(0.0, radial - self.grasp_enclosure_radius)
        axial_norm = axial_excess / max(half_gap, 1e-6)
        radial_norm = radial_excess / max(self.grasp_enclosure_radius, 1e-6)
        penalty = -self.grasp_enclosure_weight * 0.5 * min(1.0, axial_norm + radial_norm)
        return penalty

    def _update_grasp_state(self, prev_ee_pos: np.ndarray, prev_keypoints: np.ndarray, prev_gripper_qpos: Optional[float]) -> None:
        """Update auto-grasp detection state after simulation step."""
        if not self.auto_grasp or not self.gripper_action_dim or not self._auto_grasp_active:
            return

        if prev_gripper_qpos is None:
            return

        current_gripper_qpos = self._get_gripper_qpos()
        if current_gripper_qpos is None:
            return

        if not self._grasp_confirmed:
            if abs(current_gripper_qpos - prev_gripper_qpos) < self.auto_grasp_stuck_tol:
                self._stuck_count += 1
            else:
                self._stuck_count = 0

            current_ee_pos = self._get_end_effector_pos()
            current_center = self._get_gripper_center_pos(current_ee_pos)
            current_keypoints = self.keypoint_tracker.get_positions()

            if 0 <= self.primary_grasp_kp_idx < len(current_keypoints):
                kp_prev = prev_keypoints[self.primary_grasp_kp_idx]
                kp_curr = current_keypoints[self.primary_grasp_kp_idx]
                prev_center = self._prev_gripper_center if self._prev_gripper_center is not None else self._get_gripper_center_pos(prev_ee_pos)
                gripper_delta = current_center - prev_center
                kp_delta = kp_curr - kp_prev
                gripper_move = float(np.linalg.norm(gripper_delta))
                kp_move = float(np.linalg.norm(kp_delta))

                if gripper_move >= self.auto_grasp_follow_min_gripper_move:
                    ratio = kp_move / (gripper_move + 1e-8)
                    cos_sim = float(np.dot(gripper_delta, kp_delta) / ((gripper_move * kp_move) + 1e-8))
                    if ratio >= self.auto_grasp_follow_ratio and cos_sim >= self.auto_grasp_follow_cos:
                        self._follow_count += 1
                    else:
                        self._follow_count = 0
                else:
                    self._follow_count = 0
            self._prev_gripper_center = current_center

            if self._stuck_count >= self.auto_grasp_stuck_steps and self._follow_count >= self.auto_grasp_follow_steps:
                self._grasp_confirmed = True
                self._gripper_hold_fraction = self._get_gripper_open_fraction()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_step = 0

        # Reset MuJoCo state completely
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Reset to home position
        self.mj_data.qpos[:] = self._home_qpos
        self.mj_data.qvel[:] = 0
        self.mj_data.ctrl[:] = 0
        self.mj_data.time = 0.0  # Explicitly reset time

        # Forward to update derived quantities
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Re-register keypoints
        self.keypoint_tracker.register_keypoints(self.keypoints_3d)

        # Reset reward adapter
        self.reward_adapter.reset()

        # Reset auto-grasp state
        self._reset_grasp_state()

        obs = self._get_obs()
        info = {"stage": self.reward_adapter.current_stage}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results."""
        self.current_step += 1

        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Pre-step state for auto-grasp detection
        prev_ee_pos = self._get_end_effector_pos()
        prev_keypoints = self.keypoint_tracker.get_positions()
        prev_gripper_qpos = self._get_gripper_qpos()

        # Auto-grasp override (if enabled)
        self._gripper_override_fraction = self._compute_auto_grasp_override(prev_ee_pos, prev_keypoints)

        # Apply action and simulate
        self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Check time limit
        if self.mj_data.time > self.max_time:
            obs = self._get_obs()
            return obs, 0.0, True, False, {"timeout": True}

        # Get observation
        obs = self._get_obs()

        # Compute reward
        ee_pos = self._get_end_effector_pos()
        keypoints = self.keypoint_tracker.get_positions()

        # Update auto-grasp detection BEFORE computing reward
        # so we know if grasp is confirmed for stage gating
        self._update_grasp_state(prev_ee_pos, prev_keypoints, prev_gripper_qpos)

        # Save current stage before compute_reward (which may advance it)
        stage_before = self.reward_adapter.current_stage

        reward, reward_info = self.reward_adapter.compute_reward(
            end_effector=ee_pos,
            keypoints=keypoints,
            action=action,
        )

        # Gate Stage 1 completion on grasp confirmation
        if self.require_grasp_for_stage1 and stage_before == 1:
            if reward_info.stage_complete and not self._grasp_confirmed:
                # Subgoal satisfied but grasp not confirmed - revert stage advancement
                self.reward_adapter.current_stage = 1
                # Remove stage bonus from reward
                reward -= self.reward_adapter.stage_bonus
                # Update reward_info to reflect actual state
                reward_info = reward_info.__class__(
                    total_reward=reward_info.total_reward - self.reward_adapter.stage_bonus,
                    stage=1,
                    subgoal_costs=reward_info.subgoal_costs,
                    path_costs=reward_info.path_costs,
                    subgoal_satisfied=reward_info.subgoal_satisfied,
                    path_satisfied=reward_info.path_satisfied,
                    stage_complete=False,  # Not complete without grasp
                    action_penalty=reward_info.action_penalty,
                    step_penalty=reward_info.step_penalty,
                )

        # Grasp enclosure reward: encourage grasp point to be between finger pads
        grasp_enclosure_reward = self._compute_grasp_enclosure_reward(keypoints)
        reward += grasp_enclosure_reward

        # Add shaped reward: guide end effector toward current stage target
        shaped_reward = self._compute_shaped_reward(ee_pos, keypoints)
        reward += shaped_reward

        # Check termination
        terminated = self.reward_adapter.check_task_complete(ee_pos, keypoints)
        truncated = self.current_step >= self.max_steps

        info = {
            "stage": self.reward_adapter.current_stage,
            "step": self.current_step,
            "subgoal_satisfied": reward_info.subgoal_satisfied,
            "path_satisfied": reward_info.path_satisfied,
            "stage_complete": reward_info.stage_complete,
            "subgoal_costs": reward_info.subgoal_costs,
            "path_costs": reward_info.path_costs,
            "shaped_reward": shaped_reward,
            "grasp_enclosure_reward": grasp_enclosure_reward,
            "ee_pos": ee_pos.tolist(),
            "auto_grasp_active": self._auto_grasp_active,
            "grasp_confirmed": self._grasp_confirmed,
            "gripper_open_fraction": self._get_gripper_open_fraction(),
        }

        return obs, reward, terminated, truncated, info

    def _compute_shaped_reward(self, ee_pos: np.ndarray, keypoints: np.ndarray) -> float:
        """Compute shaped reward to guide end effector toward target.

        This provides continuous feedback even when far from the goal.
        """
        stage = self.reward_adapter.current_stage
        shaped_weight = self.config.get("shaped_reward_weight", 0.1)

        if stage == 1:
            # Stage 1: Move toward grasp point
            grasp_kp_idx = self.generation_result.grasp_keypoints[0]
            if grasp_kp_idx >= 0 and grasp_kp_idx < len(keypoints):
                target = keypoints[grasp_kp_idx]
                distance = np.linalg.norm(ee_pos - target)

                # Simple distance-based reward with bonus for being close
                if distance < 0.05:  # Within 5cm - bonus
                    return shaped_weight * (1.0 - distance * 10)
                else:
                    return shaped_weight * (-distance)  # Linear penalty

        elif stage == 2:
            # Stage 2: Lift the object (reward for height)
            grasp_kp_idx = self.generation_result.grasp_keypoints[0]
            if grasp_kp_idx >= 0 and grasp_kp_idx < len(keypoints):
                object_height = keypoints[grasp_kp_idx][2]
                target_height = 0.6  # From constraint
                height_diff = target_height - object_height
                return shaped_weight * (-height_diff if height_diff > 0 else 0.1)

        elif stage == 3:
            # Stage 3: Move toward pot (keypoints 5-9)
            pot_center = np.mean(keypoints[5:10], axis=0)
            grasp_kp_idx = self.generation_result.grasp_keypoints[0]
            if grasp_kp_idx >= 0:
                bottle_pos = keypoints[grasp_kp_idx]
                horizontal_dist = np.linalg.norm(bottle_pos[:2] - pot_center[:2])
                return shaped_weight * (np.exp(-horizontal_dist * 10) - 0.5)

        return 0.0

    def render(self):
        """Render environment."""
        if self.render_mode == "rgb_array" and self._renderer:
            self._renderer.update_scene(self.mj_data)
            return self._renderer.render()
        return None

    def close(self):
        """Clean up resources."""
        if self._renderer:
            self._renderer.close()
            self._renderer = None


def make_standalone_env(
    config_path: str,
    constraints_dir: str,
    render_mode: Optional[str] = None,
) -> StandaloneVLMRLEnv:
    """
    Factory function to create StandaloneVLMRLEnv.

    Args:
        config_path: Path to task config YAML
        constraints_dir: Directory with generated constraints
        render_mode: Render mode

    Returns:
        Configured environment
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve MJCF path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mjcf_path = config.get("mjcf_file_path", "")
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.join(base_dir, mjcf_path)

    return StandaloneVLMRLEnv(
        mjcf_path=mjcf_path,
        constraints_dir=constraints_dir,
        config=config,
        render_mode=render_mode,
    )
