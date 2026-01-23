"""
KeypointTracker: Track keypoint positions during simulation.

Uses relative coordinate binding (like ReKep) to track keypoints
as objects move in the simulation.
"""

import numpy as np
import mujoco
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KeypointBinding:
    """Binding of a keypoint to a body."""
    body_id: int                    # MuJoCo body ID
    body_name: str                  # Body name for reference
    local_position: np.ndarray      # Position in body's local frame
    initial_world_position: np.ndarray  # Initial world position


class KeypointTracker:
    """
    Track keypoints by binding them to bodies and using relative coordinates.

    This allows keypoints to move with objects during manipulation,
    similar to ReKep's approach in OmniGibson.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ):
        """
        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.bindings: List[KeypointBinding] = []
        self._registered = False

    def register_keypoints(
        self,
        keypoints: np.ndarray,
        exclude_body_names: Optional[List[str]] = None,
    ) -> None:
        """
        Register keypoints by binding each to the nearest body.

        Args:
            keypoints: (N, 3) initial keypoint positions in world frame
            exclude_body_names: Body names to exclude from binding (e.g., robot parts)
        """
        if exclude_body_names is None:
            exclude_body_names = []

        self.bindings = []

        for kp_idx, kp_world in enumerate(keypoints):
            # Find nearest body
            best_body_id = -1
            best_distance = float('inf')
            best_body_name = ""

            for body_id in range(self.mj_model.nbody):
                body_name = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )

                # Skip excluded bodies
                if body_name in exclude_body_names:
                    continue
                if any(excl in body_name.lower() for excl in ['robot', 'base', 'link', 'gripper', 'world']):
                    continue

                # Get body position
                body_pos = self.mj_data.xpos[body_id]
                distance = np.linalg.norm(kp_world - body_pos)

                if distance < best_distance:
                    best_distance = distance
                    best_body_id = body_id
                    best_body_name = body_name

            if best_body_id < 0:
                # Fallback: bind to world body (static)
                best_body_id = 0
                best_body_name = "world"

            # Calculate local position in body frame
            body_pos = self.mj_data.xpos[best_body_id]
            body_rot = self.mj_data.xmat[best_body_id].reshape(3, 3)

            # Transform world point to body local frame
            local_pos = body_rot.T @ (kp_world - body_pos)

            self.bindings.append(KeypointBinding(
                body_id=best_body_id,
                body_name=best_body_name,
                local_position=local_pos.copy(),
                initial_world_position=kp_world.copy(),
            ))

        self._registered = True

    def get_positions(self) -> np.ndarray:
        """
        Get current world positions of all tracked keypoints.

        Returns:
            (N, 3) array of keypoint positions in world frame
        """
        if not self._registered:
            raise RuntimeError("Keypoints not registered. Call register_keypoints first.")

        positions = []

        for binding in self.bindings:
            # Get current body pose
            body_pos = self.mj_data.xpos[binding.body_id]
            body_rot = self.mj_data.xmat[binding.body_id].reshape(3, 3)

            # Transform local position to world frame
            world_pos = body_pos + body_rot @ binding.local_position
            positions.append(world_pos)

        return np.array(positions)

    def get_position(self, index: int) -> np.ndarray:
        """
        Get current world position of a specific keypoint.

        Args:
            index: Keypoint index

        Returns:
            (3,) position in world frame
        """
        if not self._registered:
            raise RuntimeError("Keypoints not registered.")

        if index < 0 or index >= len(self.bindings):
            raise IndexError(f"Keypoint index {index} out of range")

        binding = self.bindings[index]
        body_pos = self.mj_data.xpos[binding.body_id]
        body_rot = self.mj_data.xmat[binding.body_id].reshape(3, 3)

        return body_pos + body_rot @ binding.local_position

    def get_binding_info(self, index: int) -> Dict:
        """
        Get binding information for a keypoint.

        Args:
            index: Keypoint index

        Returns:
            Dictionary with binding details
        """
        if index < 0 or index >= len(self.bindings):
            raise IndexError(f"Keypoint index {index} out of range")

        binding = self.bindings[index]
        return {
            "body_id": binding.body_id,
            "body_name": binding.body_name,
            "local_position": binding.local_position.copy(),
            "initial_world_position": binding.initial_world_position.copy(),
            "current_world_position": self.get_position(index).copy(),
        }

    def rebind_keypoint(
        self,
        index: int,
        new_body_name: str,
    ) -> None:
        """
        Rebind a keypoint to a different body.

        Useful when an object is grasped and should now move with the gripper.

        Args:
            index: Keypoint index
            new_body_name: Name of body to bind to
        """
        if index < 0 or index >= len(self.bindings):
            raise IndexError(f"Keypoint index {index} out of range")

        # Get new body ID
        new_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, new_body_name
        )
        if new_body_id < 0:
            raise ValueError(f"Body '{new_body_name}' not found")

        # Get current world position
        current_world_pos = self.get_position(index)

        # Calculate new local position
        new_body_pos = self.mj_data.xpos[new_body_id]
        new_body_rot = self.mj_data.xmat[new_body_id].reshape(3, 3)
        new_local_pos = new_body_rot.T @ (current_world_pos - new_body_pos)

        # Update binding
        self.bindings[index] = KeypointBinding(
            body_id=new_body_id,
            body_name=new_body_name,
            local_position=new_local_pos,
            initial_world_position=self.bindings[index].initial_world_position,
        )

    def get_keypoints_by_body(self, body_name: str) -> List[int]:
        """
        Get indices of keypoints bound to a specific body.

        Args:
            body_name: Name of body

        Returns:
            List of keypoint indices
        """
        return [
            i for i, binding in enumerate(self.bindings)
            if binding.body_name == body_name
        ]

    @property
    def num_keypoints(self) -> int:
        """Number of registered keypoints."""
        return len(self.bindings)

    def save_state(self) -> Dict:
        """
        Save tracker state for serialization.

        Returns:
            Dictionary containing all binding information
        """
        return {
            "bindings": [
                {
                    "body_id": b.body_id,
                    "body_name": b.body_name,
                    "local_position": b.local_position.tolist(),
                    "initial_world_position": b.initial_world_position.tolist(),
                }
                for b in self.bindings
            ]
        }

    def load_state(self, state: Dict) -> None:
        """
        Load tracker state from serialized data.

        Args:
            state: Dictionary from save_state()
        """
        self.bindings = [
            KeypointBinding(
                body_id=b["body_id"],
                body_name=b["body_name"],
                local_position=np.array(b["local_position"]),
                initial_world_position=np.array(b["initial_world_position"]),
            )
            for b in state["bindings"]
        ]
        self._registered = True
