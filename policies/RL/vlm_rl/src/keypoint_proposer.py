"""
KeypointProposer: Generate candidate keypoints from MuJoCo scene geometry.

Unlike ReKep which uses DINOv2 features from images, we sample directly from
MuJoCo geom surfaces since we have access to exact geometry in simulation.
"""

import numpy as np
import mujoco
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class KeypointProposalResult:
    """Result of keypoint proposal."""
    keypoints_3d: np.ndarray           # (N, 3) world coordinates
    keypoint_object_ids: List[int]     # which object each keypoint belongs to
    object_names: List[str]            # names of objects
    object_keypoint_ranges: Dict[str, Tuple[int, int]]  # {obj_name: (start_idx, end_idx)}


class KeypointProposer:
    """
    Generate candidate keypoints by sampling from MuJoCo geom surfaces.

    For each object body, we:
    1. Find all geoms belonging to that body
    2. Sample points from the geom surfaces
    3. Apply Farthest Point Sampling (FPS) to get diverse keypoints
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        points_per_object: int = 5,
        min_dist_between_keypoints: float = 0.02,
        surface_sample_count: int = 500,
    ):
        """
        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data
            points_per_object: Number of keypoints to generate per object
            min_dist_between_keypoints: Minimum distance between keypoints (meters)
            surface_sample_count: Number of points to sample from each geom surface
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.points_per_object = points_per_object
        self.min_dist = min_dist_between_keypoints
        self.surface_sample_count = surface_sample_count

    def propose(
        self,
        object_body_names: List[str],
        exclude_keywords: Optional[List[str]] = None,
    ) -> KeypointProposalResult:
        """
        Generate candidate keypoints for specified objects.

        Args:
            object_body_names: List of MuJoCo body names to generate keypoints for
            exclude_keywords: Body names containing these keywords are skipped

        Returns:
            KeypointProposalResult with keypoints and metadata
        """
        if exclude_keywords is None:
            exclude_keywords = ['robot', 'floor', 'wall', 'ground']

        all_keypoints = []
        keypoint_object_ids = []
        object_keypoint_ranges = {}
        valid_object_names = []

        for obj_idx, body_name in enumerate(object_body_names):
            # Skip excluded objects
            if any(kw in body_name.lower() for kw in exclude_keywords):
                continue

            # Get body ID
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                print(f"Warning: Body '{body_name}' not found, skipping")
                continue

            # Sample surface points from all geoms of this body
            surface_points = self._sample_body_surface(body_id)

            if len(surface_points) < self.points_per_object:
                print(f"Warning: Not enough surface points for '{body_name}'")
                continue

            # Apply FPS to get diverse keypoints
            keypoints = self._farthest_point_sampling(
                surface_points, self.points_per_object
            )

            # Record range
            start_idx = len(all_keypoints)
            all_keypoints.extend(keypoints)
            end_idx = len(all_keypoints)

            object_keypoint_ranges[body_name] = (start_idx, end_idx)
            keypoint_object_ids.extend([obj_idx] * len(keypoints))
            valid_object_names.append(body_name)

        return KeypointProposalResult(
            keypoints_3d=np.array(all_keypoints) if all_keypoints else np.zeros((0, 3)),
            keypoint_object_ids=keypoint_object_ids,
            object_names=valid_object_names,
            object_keypoint_ranges=object_keypoint_ranges,
        )

    def _sample_body_surface(self, body_id: int) -> np.ndarray:
        """Sample points from all geom surfaces of a body."""
        all_points = []

        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_bodyid[geom_id] != body_id:
                continue

            geom_type = self.mj_model.geom_type[geom_id]
            geom_size = self.mj_model.geom_size[geom_id].copy()
            geom_pos = self.mj_data.geom_xpos[geom_id].copy()
            geom_mat = self.mj_data.geom_xmat[geom_id].reshape(3, 3)

            # Sample based on geom type
            local_points = self._sample_geom_surface(geom_type, geom_size)

            # Transform to world coordinates
            world_points = (geom_mat @ local_points.T).T + geom_pos
            all_points.append(world_points)

        if not all_points:
            return np.zeros((0, 3))

        return np.vstack(all_points)

    def _sample_geom_surface(
        self,
        geom_type: int,
        geom_size: np.ndarray
    ) -> np.ndarray:
        """Sample points from a geom surface in local coordinates."""
        n = self.surface_sample_count

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return self._sample_box_surface(geom_size, n)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return self._sample_sphere_surface(geom_size[0], n)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return self._sample_cylinder_surface(geom_size, n)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return self._sample_capsule_surface(geom_size, n)
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return self._sample_ellipsoid_surface(geom_size, n)
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # For mesh, sample from bounding box as approximation
            # In production, should sample from actual mesh vertices
            return self._sample_box_surface(geom_size, n)
        else:
            # Default: sample from bounding sphere
            max_size = np.max(geom_size) if len(geom_size) > 0 else 0.05
            return self._sample_sphere_surface(max_size, n)

    def _sample_box_surface(self, half_sizes: np.ndarray, n: int) -> np.ndarray:
        """Sample points uniformly from box surface."""
        hx, hy, hz = half_sizes[:3]

        # Calculate face areas
        areas = np.array([
            hy * hz,  # +x, -x faces
            hx * hz,  # +y, -y faces
            hx * hy,  # +z, -z faces
        ]) * 2  # two faces per axis

        total_area = 2 * np.sum(areas)
        probs = np.repeat(areas, 2) / total_area

        points = []
        for _ in range(n):
            # Choose a face
            face = np.random.choice(6, p=probs)
            axis = face // 2
            sign = 1 if face % 2 == 0 else -1

            # Sample on that face
            pt = np.zeros(3)
            pt[axis] = sign * half_sizes[axis]

            other_axes = [i for i in range(3) if i != axis]
            for ax in other_axes:
                pt[ax] = np.random.uniform(-half_sizes[ax], half_sizes[ax])

            points.append(pt)

        return np.array(points)

    def _sample_sphere_surface(self, radius: float, n: int) -> np.ndarray:
        """Sample points uniformly from sphere surface."""
        # Use spherical coordinates with uniform distribution
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        x = radius * sin_theta * np.cos(phi)
        y = radius * sin_theta * np.sin(phi)
        z = radius * cos_theta

        return np.stack([x, y, z], axis=1)

    def _sample_cylinder_surface(self, size: np.ndarray, n: int) -> np.ndarray:
        """Sample from cylinder surface (radius=size[0], half_height=size[1])."""
        radius = size[0]
        half_height = size[1]

        # Allocate points between lateral surface and caps
        lateral_area = 2 * np.pi * radius * 2 * half_height
        cap_area = 2 * np.pi * radius ** 2
        total_area = lateral_area + cap_area

        n_lateral = int(n * lateral_area / total_area)
        n_caps = n - n_lateral

        points = []

        # Lateral surface
        theta = np.random.uniform(0, 2 * np.pi, n_lateral)
        z = np.random.uniform(-half_height, half_height, n_lateral)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append(np.stack([x, y, z], axis=1))

        # Caps
        for _ in range(n_caps):
            r = radius * np.sqrt(np.random.uniform())
            theta = np.random.uniform(0, 2 * np.pi)
            z = half_height if np.random.random() > 0.5 else -half_height
            points.append([[r * np.cos(theta), r * np.sin(theta), z]])

        return np.vstack(points)

    def _sample_capsule_surface(self, size: np.ndarray, n: int) -> np.ndarray:
        """Sample from capsule surface (radius=size[0], half_length=size[1])."""
        radius = size[0]
        half_length = size[1]

        # Capsule = cylinder + two hemispheres
        cyl_area = 2 * np.pi * radius * 2 * half_length
        sphere_area = 4 * np.pi * radius ** 2
        total_area = cyl_area + sphere_area

        n_cyl = int(n * cyl_area / total_area)
        n_sphere = n - n_cyl

        points = []

        # Cylinder part
        theta = np.random.uniform(0, 2 * np.pi, n_cyl)
        z = np.random.uniform(-half_length, half_length, n_cyl)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append(np.stack([x, y, z], axis=1))

        # Hemisphere caps
        sphere_points = self._sample_sphere_surface(radius, n_sphere)
        sphere_points[:, 2] = np.abs(sphere_points[:, 2])  # upper hemisphere

        # Split between top and bottom
        half = len(sphere_points) // 2
        top_pts = sphere_points[:half].copy()
        top_pts[:, 2] += half_length

        bottom_pts = sphere_points[half:].copy()
        bottom_pts[:, 2] = -bottom_pts[:, 2] - half_length

        points.append(top_pts)
        points.append(bottom_pts)

        return np.vstack(points)

    def _sample_ellipsoid_surface(self, half_sizes: np.ndarray, n: int) -> np.ndarray:
        """Sample from ellipsoid surface."""
        # Sample from unit sphere and scale
        sphere_pts = self._sample_sphere_surface(1.0, n)
        return sphere_pts * half_sizes[:3]

    def _farthest_point_sampling(
        self,
        points: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """
        Farthest Point Sampling (FPS) for diverse point selection.

        Args:
            points: (N, 3) candidate points
            num_samples: number of points to select

        Returns:
            (num_samples, 3) selected points
        """
        if len(points) <= num_samples:
            return points

        n = len(points)
        selected_indices = [np.random.randint(n)]
        distances = np.full(n, np.inf)

        for _ in range(num_samples - 1):
            # Update distances to nearest selected point
            last_selected = points[selected_indices[-1]]
            dist_to_last = np.linalg.norm(points - last_selected, axis=1)
            distances = np.minimum(distances, dist_to_last)

            # Select point with maximum distance to any selected point
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)

        return points[selected_indices]

    def merge_close_keypoints(
        self,
        keypoints: np.ndarray,
        min_distance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge keypoints that are too close together.

        Args:
            keypoints: (N, 3) keypoints
            min_distance: minimum distance threshold (uses self.min_dist if None)

        Returns:
            merged_keypoints: (M, 3) merged keypoints
            mapping: (N,) index mapping from original to merged
        """
        if min_distance is None:
            min_distance = self.min_dist

        if len(keypoints) == 0:
            return keypoints, np.array([])

        # Simple greedy merging
        merged = []
        mapping = np.zeros(len(keypoints), dtype=int)

        for i, kp in enumerate(keypoints):
            merged_idx = -1
            for j, mkp in enumerate(merged):
                if np.linalg.norm(kp - mkp) < min_distance:
                    merged_idx = j
                    break

            if merged_idx >= 0:
                mapping[i] = merged_idx
            else:
                mapping[i] = len(merged)
                merged.append(kp)

        return np.array(merged), mapping
