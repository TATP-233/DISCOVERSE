"""
ConstraintRewardAdapter: Convert VLM-generated constraints to RL rewards.

Takes constraint cost values and converts them to reward signals for PPO training.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardInfo:
    """Detailed reward computation information."""
    total_reward: float
    stage: int
    subgoal_costs: List[float]
    path_costs: List[float]
    subgoal_satisfied: bool
    path_satisfied: bool
    stage_complete: bool
    action_penalty: float
    step_penalty: float


class ConstraintRewardAdapter:
    """
    Convert constraint costs to RL rewards.

    Supports multiple reward shaping strategies and multi-stage task progression.
    """

    def __init__(
        self,
        constraints: Dict[int, Dict[str, List[Callable]]],
        num_stages: int,
        grasp_keypoints: List[int],
        release_keypoints: List[int],
        # Reward shaping parameters
        reward_type: str = "negative",  # "negative", "exponential", "sparse"
        subgoal_weight: float = 2.0,
        path_weight: float = 1.0,
        action_penalty_weight: float = 0.1,
        step_penalty_weight: float = 0.001,
        # Stage progression
        subgoal_threshold: float = 0.02,
        path_threshold: float = 0.1,  # Threshold for path constraint satisfaction
        stage_bonus: float = 10.0,
        task_complete_bonus: float = 100.0,
    ):
        """
        Args:
            constraints: {stage: {"subgoal": [fn, ...], "path": [fn, ...]}}
            num_stages: Total number of stages
            grasp_keypoints: Keypoint to grasp at each stage (-1 = none)
            release_keypoints: Keypoint to release at each stage (-1 = none)
            reward_type: How to convert cost to reward
            subgoal_weight: Weight for subgoal constraints
            path_weight: Weight for path constraints
            action_penalty_weight: Penalty for large actions
            step_penalty_weight: Penalty per step (encourages efficiency)
            subgoal_threshold: Cost threshold for considering subgoal satisfied
            path_threshold: Cost threshold for considering path constraint satisfied
            stage_bonus: Bonus reward for completing a stage
            task_complete_bonus: Bonus for completing entire task
        """
        self.constraints = constraints
        self.num_stages = num_stages
        self.grasp_keypoints = grasp_keypoints
        self.release_keypoints = release_keypoints

        self.reward_type = reward_type
        self.subgoal_weight = subgoal_weight
        self.path_weight = path_weight
        self.action_penalty_weight = action_penalty_weight
        self.step_penalty_weight = step_penalty_weight

        self.subgoal_threshold = subgoal_threshold
        self.path_threshold = path_threshold
        self.stage_bonus = stage_bonus
        self.task_complete_bonus = task_complete_bonus

        # State tracking
        self.current_stage = 1
        self.step_count = 0
        self._prev_subgoal_cost = float('inf')

    def reset(self) -> None:
        """Reset adapter state for new episode."""
        self.current_stage = 1
        self.step_count = 0
        self._prev_subgoal_cost = float('inf')

    def compute_reward(
        self,
        end_effector: np.ndarray,
        keypoints: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> Tuple[float, RewardInfo]:
        """
        Compute reward for current state.

        Args:
            end_effector: (3,) end effector position
            keypoints: (N, 3) keypoint positions
            action: Optional action array for action penalty

        Returns:
            reward: Scalar reward value
            info: Detailed RewardInfo
        """
        self.step_count += 1
        stage = self.current_stage

        # Get constraints for current stage
        stage_constraints = self.constraints.get(stage, {"subgoal": [], "path": []})

        # Compute subgoal costs
        subgoal_costs = []
        for fn in stage_constraints.get("subgoal", []):
            try:
                cost = fn(end_effector, keypoints)
                subgoal_costs.append(float(cost))
            except Exception as e:
                print(f"Warning: Subgoal constraint error: {e}")
                subgoal_costs.append(1.0)

        # Compute path costs
        path_costs = []
        for fn in stage_constraints.get("path", []):
            try:
                cost = fn(end_effector, keypoints)
                path_costs.append(float(cost))
            except Exception as e:
                print(f"Warning: Path constraint error: {e}")
                path_costs.append(1.0)

        # Aggregate costs
        total_subgoal_cost = sum(subgoal_costs) if subgoal_costs else 0.0
        total_path_cost = sum(path_costs) if path_costs else 0.0

        # Check if constraints are satisfied
        subgoal_satisfied = all(c <= self.subgoal_threshold for c in subgoal_costs) if subgoal_costs else True
        path_satisfied = all(c <= self.path_threshold for c in path_costs) if path_costs else True

        # Convert costs to rewards
        subgoal_reward = self._cost_to_reward(total_subgoal_cost) * self.subgoal_weight
        path_reward = self._cost_to_reward(total_path_cost) * self.path_weight

        # Progress reward (encourage improvement)
        progress_reward = 0.0
        if self._prev_subgoal_cost < float('inf') and total_subgoal_cost < self._prev_subgoal_cost:
            progress_reward = (self._prev_subgoal_cost - total_subgoal_cost) * 0.5
        self._prev_subgoal_cost = total_subgoal_cost

        # Action penalty
        action_penalty = 0.0
        if action is not None:
            action_penalty = -self.action_penalty_weight * np.mean(np.abs(action))

        # Step penalty
        step_penalty = -self.step_penalty_weight

        # Stage completion bonus
        stage_complete = False
        stage_bonus = 0.0
        if subgoal_satisfied and path_satisfied:
            stage_complete = True
            stage_bonus = self.stage_bonus

            # Move to next stage
            if self.current_stage < self.num_stages:
                self.current_stage += 1
                self._prev_subgoal_cost = float('inf')

        # Task completion bonus
        task_bonus = 0.0
        if stage_complete and stage == self.num_stages:
            task_bonus = self.task_complete_bonus

        # Total reward
        total_reward = (
            subgoal_reward +
            path_reward +
            progress_reward +
            action_penalty +
            step_penalty +
            stage_bonus +
            task_bonus
        )

        info = RewardInfo(
            total_reward=total_reward,
            stage=stage,
            subgoal_costs=subgoal_costs,
            path_costs=path_costs,
            subgoal_satisfied=subgoal_satisfied,
            path_satisfied=path_satisfied,
            stage_complete=stage_complete,
            action_penalty=action_penalty,
            step_penalty=step_penalty,
        )

        return total_reward, info

    def _cost_to_reward(self, cost: float) -> float:
        """Convert constraint cost to reward value."""
        if self.reward_type == "negative":
            # Simple negation: reward = -cost
            return -cost

        elif self.reward_type == "exponential":
            # Exponential decay: high reward when cost is low
            return np.exp(-cost)

        elif self.reward_type == "sparse":
            # Sparse: reward only when satisfied
            return 1.0 if cost <= 0.0 else 0.0

        elif self.reward_type == "tanh":
            # Smooth bounded reward
            return 1.0 - np.tanh(cost)

        else:
            return -cost

    def check_stage_complete(
        self,
        end_effector: np.ndarray,
        keypoints: np.ndarray,
    ) -> bool:
        """
        Check if current stage is complete.

        Args:
            end_effector: (3,) end effector position
            keypoints: (N, 3) keypoint positions

        Returns:
            True if stage is complete
        """
        stage = self.current_stage
        stage_constraints = self.constraints.get(stage, {"subgoal": [], "path": []})

        # Check all subgoal constraints
        for fn in stage_constraints.get("subgoal", []):
            try:
                cost = fn(end_effector, keypoints)
                if cost > self.subgoal_threshold:
                    return False
            except:
                return False

        return True

    def check_task_complete(
        self,
        end_effector: np.ndarray,
        keypoints: np.ndarray,
    ) -> bool:
        """
        Check if entire task is complete.

        Args:
            end_effector: (3,) end effector position
            keypoints: (N, 3) keypoint positions

        Returns:
            True if task is complete
        """
        # Check if we're at the last stage and it's complete
        if self.current_stage < self.num_stages:
            return False

        return self.check_stage_complete(end_effector, keypoints)

    def get_grasp_keypoint(self) -> int:
        """Get keypoint index to grasp at current stage (-1 if none)."""
        if self.current_stage <= len(self.grasp_keypoints):
            return self.grasp_keypoints[self.current_stage - 1]
        return -1

    def get_release_keypoint(self) -> int:
        """Get keypoint index to release at current stage (-1 if none)."""
        if self.current_stage <= len(self.release_keypoints):
            return self.release_keypoints[self.current_stage - 1]
        return -1

    def get_reward_breakdown(self) -> Dict[str, float]:
        """
        Get dictionary of reward component weights.

        Useful for logging and debugging.
        """
        return {
            "subgoal_weight": self.subgoal_weight,
            "path_weight": self.path_weight,
            "action_penalty_weight": self.action_penalty_weight,
            "step_penalty_weight": self.step_penalty_weight,
            "stage_bonus": self.stage_bonus,
            "task_complete_bonus": self.task_complete_bonus,
        }

    def set_stage(self, stage: int) -> None:
        """
        Manually set current stage.

        Args:
            stage: Stage number (1-indexed)
        """
        if stage < 1 or stage > self.num_stages:
            raise ValueError(f"Stage must be between 1 and {self.num_stages}")
        self.current_stage = stage
        self._prev_subgoal_cost = float('inf')


class SimpleRewardAdapter:
    """
    Simplified reward adapter for single-stage tasks.

    Just converts constraint costs to rewards without stage management.
    """

    def __init__(
        self,
        constraints: List[Callable],
        weights: Optional[List[float]] = None,
        action_penalty_weight: float = 0.1,
    ):
        """
        Args:
            constraints: List of constraint functions
            weights: Optional weights for each constraint
            action_penalty_weight: Penalty for large actions
        """
        self.constraints = constraints
        self.weights = weights or [1.0] * len(constraints)
        self.action_penalty_weight = action_penalty_weight

    def compute_reward(
        self,
        end_effector: np.ndarray,
        keypoints: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """
        Compute reward from constraints.

        Returns:
            reward: Total reward
            info: Dictionary with cost breakdown
        """
        costs = []
        for fn in self.constraints:
            try:
                cost = fn(end_effector, keypoints)
                costs.append(float(cost))
            except Exception as e:
                costs.append(1.0)

        # Weighted sum of costs
        weighted_cost = sum(w * c for w, c in zip(self.weights, costs))

        # Convert to reward (using tanh for bounded output)
        reward = 1.0 - np.tanh(weighted_cost)

        # Action penalty
        if action is not None:
            reward -= self.action_penalty_weight * np.mean(np.abs(action))

        info = {
            "costs": costs,
            "weighted_cost": weighted_cost,
            "satisfied": all(c <= 0 for c in costs),
        }

        return reward, info
