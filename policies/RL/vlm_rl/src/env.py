"""
VLMRLEnv: Gymnasium environment for VLM-guided reinforcement learning.

Integrates keypoint tracking and VLM-generated constraints for reward computation.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple, Any
import os
import yaml

from .keypoint_proposer import KeypointProposer, KeypointProposalResult
from .keypoint_tracker import KeypointTracker
from .annotated_renderer import AnnotatedRenderer
from .constraint_generator import ConstraintGenerator, ConstraintLoader, GenerationResult
from .reward_adapter import ConstraintRewardAdapter


class VLMRLEnv(gymnasium.Env):
    """
    Gymnasium environment with VLM-generated constraint rewards.

    This environment:
    1. Proposes keypoints from MuJoCo scene geometry
    2. Uses VLM to generate task constraints from annotated images
    3. Tracks keypoints during simulation
    4. Computes rewards from constraint satisfaction
    """

    def __init__(
        self,
        task_base,
        config: Dict[str, Any],
        constraints_dir: Optional[str] = None,
        render_mode: bool = False,
    ):
        """
        Args:
            task_base: DISCOVERSE task base (e.g., SimNode)
            config: Configuration dictionary containing:
                - object_bodies: List of object body names
                - end_effector_body: End effector body name
                - instruction: Task instruction (for VLM)
                - max_steps: Maximum steps per episode
                - points_per_object: Keypoints per object
                - reward_config: Reward adapter configuration
            constraints_dir: Directory containing pre-generated constraints
                            (if None, will generate on first reset)
            render_mode: Whether to enable rendering
        """
        super().__init__()

        self.task_base = task_base
        self.config = config
        self.constraints_dir = constraints_dir
        self.render_mode = render_mode

        # Get MuJoCo model and data
        self.mj_model = task_base.mj_model
        self.mj_data = task_base.mj_data

        # Configuration
        self.object_bodies = config.get("object_bodies", [])
        self.end_effector_body = config.get("end_effector_body", "rgt_arm_link6")
        self.instruction = config.get("instruction", "")
        self.max_steps = config.get("max_steps", 1000)
        self.max_time = config.get("max_time", 20.0)

        # Keypoint configuration
        self.points_per_object = config.get("points_per_object", 5)

        # Initialize components
        self._init_keypoint_proposer()
        self._init_renderer()

        # These will be initialized on first reset
        self.keypoint_tracker: Optional[KeypointTracker] = None
        self.reward_adapter: Optional[ConstraintRewardAdapter] = None
        self.keypoints_3d: Optional[np.ndarray] = None
        self.generation_result: Optional[GenerationResult] = None

        # Action space: robot joint control
        ctrl_range = self.mj_model.actuator_ctrlrange.astype(np.float32)
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0],
            high=ctrl_range[:, 1],
            dtype=np.float32
        )

        # Observation space will be set after keypoint initialization
        # For now, use a placeholder
        self._obs_dim = self.mj_model.nq + self.mj_model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim + 100,),  # Placeholder, will be updated
            dtype=np.float32
        )

        # State
        self.current_step = 0
        self.reward_info = {}
        self._initialized = False

    def _init_keypoint_proposer(self):
        """Initialize keypoint proposer."""
        self.keypoint_proposer = KeypointProposer(
            mj_model=self.mj_model,
            mj_data=self.mj_data,
            points_per_object=self.points_per_object,
        )

    def _init_renderer(self):
        """Initialize annotated renderer."""
        camera_name = self.config.get("camera_name", None)
        self.renderer = AnnotatedRenderer(
            mj_model=self.mj_model,
            mj_data=self.mj_data,
            width=self.config.get("image_width", 640),
            height=self.config.get("image_height", 480),
            camera_name=camera_name,
        )

    def _init_keypoints_and_constraints(self):
        """Initialize keypoints and load/generate constraints."""
        # Propose keypoints
        proposal = self.keypoint_proposer.propose(self.object_bodies)
        self.keypoints_3d = proposal.keypoints_3d
        self.keypoint_proposal = proposal

        # Initialize tracker
        self.keypoint_tracker = KeypointTracker(self.mj_model, self.mj_data)
        self.keypoint_tracker.register_keypoints(self.keypoints_3d)

        # Load or generate constraints
        if self.constraints_dir and os.path.exists(self.constraints_dir):
            # Load existing constraints
            self._load_constraints()
        else:
            # Generate new constraints using VLM
            self._generate_constraints()

        # Update observation space
        num_keypoints = len(self.keypoints_3d)
        # obs = qpos + qvel + end_effector_pos + keypoints_flat
        self._obs_dim = self.mj_model.nq + self.mj_model.nv + 3 + num_keypoints * 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

        self._initialized = True

    def _load_constraints(self):
        """Load pre-generated constraints from directory."""
        # Load generation result
        import json
        result_path = os.path.join(self.constraints_dir, "generation_result.json")

        if os.path.exists(result_path):
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
        else:
            # Try to load from constraint files directly
            loader = ConstraintLoader()
            constraints = loader.load_from_directory(self.constraints_dir)

            # Create minimal generation result
            num_stages = max(constraints.keys()) if constraints else 1
            self.generation_result = GenerationResult(
                num_stages=num_stages,
                grasp_keypoints=[-1] * num_stages,
                release_keypoints=[-1] * num_stages,
                subgoal_constraints={},
                path_constraints={},
                raw_output="",
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

    def _generate_constraints(self):
        """Generate constraints using VLM."""
        # Create output directory
        if self.constraints_dir is None:
            self.constraints_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs",
                "generated_constraints"
            )
        os.makedirs(self.constraints_dir, exist_ok=True)

        # Render annotated image
        annotated_image = self.renderer.render_with_keypoints_by_object(
            self.keypoints_3d,
            self.keypoint_proposal.object_keypoint_ranges,
        )

        # Generate keypoint description
        keypoint_desc = self.renderer.generate_keypoint_description(
            self.keypoints_3d,
            self.keypoint_proposal.object_keypoint_ranges,
        )

        # Call VLM
        generator = ConstraintGenerator(
            model=self.config.get("vlm_model", "gpt-4o"),
            api_key=self.config.get("openai_api_key"),
        )

        self.generation_result = generator.generate(
            image=annotated_image,
            instruction=self.instruction,
            keypoint_description=keypoint_desc,
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation, info
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Reset task base
        self.task_base.reset()

        # Apply domain randomization if available
        if hasattr(self.task_base, 'domain_randomization'):
            self.task_base.domain_randomization()

        # Initialize keypoints and constraints on first reset
        if not self._initialized:
            self._init_keypoints_and_constraints()
        else:
            # Re-register keypoints (positions may have changed due to randomization)
            self.keypoint_tracker.register_keypoints(self.keypoints_3d)

        # Reset reward adapter
        if self.reward_adapter:
            self.reward_adapter.reset()

        observation = self._get_obs()
        info = {"stage": self.reward_adapter.current_stage if self.reward_adapter else 1}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return results.

        Args:
            action: Joint control action

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Check time limit
        if self.mj_data.time > self.max_time:
            obs = self._get_obs()
            return obs, 0.0, True, False, {"timeout": True}

        # Execute action
        try:
            self.task_base.step(action)
        except Exception as e:
            print(f"Step failed: {e}")
            obs = self._get_obs()
            return obs, 0.0, True, False, {"error": str(e)}

        # Get observation
        observation = self._get_obs()

        # Compute reward
        reward, reward_info = self._compute_reward(action)

        # Check termination
        terminated = self._check_success()
        truncated = self.current_step >= self.max_steps

        # Build info dict
        info = {
            "stage": self.reward_adapter.current_stage if self.reward_adapter else 1,
            "step": self.current_step,
        }
        info.update(self.reward_info)

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        # Joint positions and velocities
        qpos = self.mj_data.qpos.copy()
        qvel = self.mj_data.qvel.copy()

        # End effector position
        ee_pos = self._get_end_effector_pos()

        # Keypoint positions (flattened)
        if self.keypoint_tracker:
            keypoints = self.keypoint_tracker.get_positions().flatten()
        else:
            keypoints = np.zeros(0)

        # Concatenate all observations
        obs = np.concatenate([qpos, qvel, ee_pos, keypoints]).astype(np.float32)

        return obs

    def _get_end_effector_pos(self) -> np.ndarray:
        """Get end effector position."""
        try:
            from discoverse.utils import get_body_tmat
            tmat = get_body_tmat(self.mj_data, self.end_effector_body)
            # Extract position (note: DISCOVERSE may use different axis convention)
            pos = np.array([tmat[0, 3], tmat[1, 3], tmat[2, 3]])
        except:
            # Fallback: get position from xpos
            body_id = self.mj_model.body(self.end_effector_body).id
            pos = self.mj_data.xpos[body_id].copy()

        return pos

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Compute reward from constraints."""
        if self.reward_adapter is None:
            return 0.0, {}

        # Get current positions
        ee_pos = self._get_end_effector_pos()
        keypoints = self.keypoint_tracker.get_positions()

        # Compute reward
        reward, info = self.reward_adapter.compute_reward(
            end_effector=ee_pos,
            keypoints=keypoints,
            action=action,
        )

        # Store for logging
        self.reward_info = {
            "rewards/total": reward,
            "rewards/stage": info.stage,
            "rewards/subgoal_satisfied": info.subgoal_satisfied,
            "rewards/path_satisfied": info.path_satisfied,
            "rewards/stage_complete": info.stage_complete,
            "info/subgoal_costs": info.subgoal_costs,
            "info/path_costs": info.path_costs,
        }

        return reward, self.reward_info

    def _check_success(self) -> bool:
        """Check if task is complete."""
        if self.reward_adapter is None:
            return False

        ee_pos = self._get_end_effector_pos()
        keypoints = self.keypoint_tracker.get_positions()

        return self.reward_adapter.check_task_complete(ee_pos, keypoints)

    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode:
            if self.keypoints_3d is not None:
                return self.renderer.render_with_keypoints(self.keypoints_3d)
            else:
                return self.renderer.render_rgb()
        return None

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'task_base') and self.task_base is not None:
            del self.task_base
            self.task_base = None

    def get_keypoint_positions(self) -> np.ndarray:
        """Get current keypoint positions (for debugging/visualization)."""
        if self.keypoint_tracker:
            return self.keypoint_tracker.get_positions()
        return np.zeros((0, 3))

    def get_constraint_info(self) -> Dict:
        """Get information about loaded constraints."""
        if self.generation_result:
            return {
                "num_stages": self.generation_result.num_stages,
                "grasp_keypoints": self.generation_result.grasp_keypoints,
                "release_keypoints": self.generation_result.release_keypoints,
                "constraints_dir": self.constraints_dir,
            }
        return {}


def make_vlm_rl_env(
    task_name: str,
    config_path: Optional[str] = None,
    constraints_dir: Optional[str] = None,
    render: bool = False,
) -> VLMRLEnv:
    """
    Factory function to create VLMRLEnv from configuration.

    Args:
        task_name: Name of the task (used to find config)
        config_path: Path to config file (optional)
        constraints_dir: Directory with pre-generated constraints
        render: Enable rendering

    Returns:
        Configured VLMRLEnv instance
    """
    # Load configuration
    if config_path is None:
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(module_dir, "configs", "tasks", f"{task_name}.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Import DISCOVERSE components
    # This assumes a specific task structure; adjust as needed
    task_module = config.get("task_module", "discoverse.examples.tasks_mmk2.kiwi_pick")
    exec(f"from {task_module} import SimNode, cfg")

    # Configure task
    cfg_updates = config.get("cfg_updates", {})
    for key, value in cfg_updates.items():
        setattr(cfg, key, value)

    cfg.headless = not render

    # Create task base
    task_base = SimNode(cfg)

    return VLMRLEnv(
        task_base=task_base,
        config=config,
        constraints_dir=constraints_dir,
        render_mode=render,
    )
