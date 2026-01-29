"""
KeypointProposer: Sample keypoints from 2D segmentation masks and back-project to 3D.

Pipeline:
1) Render segmentation + depth from MuJoCo
2) Sample 2D keypoints on object masks (center + FPS)
3) Back-project to 3D using depth + camera intrinsics/extrinsics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mujoco


@dataclass
class KeypointProposalResult:
    """Result of keypoint proposal."""
    keypoints_3d: np.ndarray           # (N, 3) world coordinates
    keypoint_object_ids: List[int]     # which object each keypoint belongs to
    object_names: List[str]            # names of objects
    object_keypoint_ranges: Dict[str, Tuple[int, int]]  # {obj_name: (start_idx, end_idx)}
    keypoints_2d: Optional[np.ndarray] = None           # (N, 2) pixel coordinates
    object_masks: Optional[Dict[str, np.ndarray]] = None  # {obj_name: (H, W) mask}


def _farthest_point_sampling_2d(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest Point Sampling for 2D points."""
    points = np.array(points)
    if len(points) <= n_samples:
        return points
    if len(points) == 0:
        return points

    n_points = len(points)
    sample_inds = np.zeros(n_samples, dtype=int)
    dists = np.full(n_points, np.inf)

    sample_inds[0] = 0

    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        dist_to_last = np.sum((points - points[last_added]) ** 2, axis=1)
        dists = np.minimum(dists, dist_to_last)
        sample_inds[i] = np.argmax(dists)

    return points[sample_inds]


def _mask_to_vertices(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to contour vertices."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)

    vertices = np.vstack([c.reshape(-1, 2) for c in contours if len(c) >= 3])
    return vertices.astype(np.float32)


def sample_keypoints_from_mask(
    mask: np.ndarray,
    num_samples: int = 5,
    include_center: bool = True,
) -> np.ndarray:
    """Sample keypoints from a segmentation mask using FPS."""
    vertices = _mask_to_vertices(mask)

    if vertices.size == 0:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        center = np.array([[xs.mean(), ys.mean()]])
        return center.astype(np.int32)

    points = vertices.copy()

    if include_center:
        center = points.mean(axis=0, keepdims=True)
        points = np.vstack([center, points])

    if points.shape[0] > num_samples:
        kps = _farthest_point_sampling_2d(points, num_samples)
    else:
        kps = points

    if kps.shape[0] > 1 and include_center:
        kps = np.vstack([kps[[0]], kps[1:][np.argsort(kps[1:, 1])]])

    return kps.astype(np.int32)


class KeypointProposer:
    """
    Propose keypoints from rendered segmentation masks and depth.

    Uses 2D mask sampling (center + FPS) and back-projects to 3D using depth.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        renderer,
        points_per_object: int = 5,
        include_center: bool = True,
        depth_search_radius: int = 6,
    ):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.renderer = renderer
        self.points_per_object = points_per_object
        self.include_center = include_center
        self.depth_search_radius = depth_search_radius

    def propose(
        self,
        object_body_names: List[str],
        exclude_keywords: Optional[List[str]] = None,
    ) -> KeypointProposalResult:
        if exclude_keywords is None:
            exclude_keywords = ['robot', 'floor', 'wall', 'ground']

        seg = self.renderer.render_segmentation()
        depth = self.renderer.render_depth()

        all_keypoints_2d = []
        all_keypoints_3d = []
        keypoint_object_ids = []
        object_keypoint_ranges = {}
        object_masks: Dict[str, np.ndarray] = {}
        valid_object_names = []

        for obj_idx, body_name in enumerate(object_body_names):
            if any(kw in body_name.lower() for kw in exclude_keywords):
                continue

            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                print(f"Warning: Body '{body_name}' not found, skipping")
                continue

            geom_ids = [
                gid for gid in range(self.mj_model.ngeom)
                if self.mj_model.geom_bodyid[gid] == body_id
            ]

            if not geom_ids:
                print(f"Warning: Body '{body_name}' has no geoms, skipping")
                continue

            geom_ids_arr = np.array(geom_ids, dtype=np.int32)
            objtype = int(mujoco.mjtObj.mjOBJ_GEOM)

            mask = (seg[:, :, 1] == objtype) & np.isin(seg[:, :, 0], geom_ids_arr)
            object_masks[body_name] = mask

            if not np.any(mask):
                print(f"Warning: No visible pixels for '{body_name}', skipping")
                continue

            kps_2d = sample_keypoints_from_mask(
                mask,
                num_samples=self.points_per_object,
                include_center=self.include_center,
            )

            if len(kps_2d) == 0:
                print(f"Warning: No keypoints sampled for '{body_name}', skipping")
                continue

            center_index = 0 if self.include_center and len(kps_2d) > 0 else None
            kps_3d = self._keypoints_2d_to_world(
                kps_2d,
                depth,
                mask,
                body_id=body_id,
                center_index=center_index,
            )

            start_idx = len(all_keypoints_2d)
            all_keypoints_2d.extend(kps_2d)
            all_keypoints_3d.extend(kps_3d)
            end_idx = len(all_keypoints_2d)

            object_keypoint_ranges[body_name] = (start_idx, end_idx)
            keypoint_object_ids.extend([obj_idx] * len(kps_2d))
            valid_object_names.append(body_name)

        return KeypointProposalResult(
            keypoints_3d=np.array(all_keypoints_3d) if all_keypoints_3d else np.zeros((0, 3)),
            keypoint_object_ids=keypoint_object_ids,
            object_names=valid_object_names,
            object_keypoint_ranges=object_keypoint_ranges,
            keypoints_2d=np.array(all_keypoints_2d) if all_keypoints_2d else np.zeros((0, 2)),
            object_masks=object_masks if object_masks else None,
        )

    def _keypoints_2d_to_world(
        self,
        keypoints_2d: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        body_id: int,
        center_index: Optional[int] = None,
    ) -> np.ndarray:
        intr = self.renderer.get_camera_intrinsics()
        cam_pos, cam_forward, cam_up, cam_right = self.renderer.get_camera_frame()

        valid = mask & np.isfinite(depth) & (depth > 1e-6)
        ys, xs = np.where(valid)
        depths = depth[ys, xs] if len(xs) > 0 else None

        body_pos = self.mj_data.xpos[body_id].copy()

        points_3d = []
        for idx, pt in enumerate(keypoints_2d):
            x, y = int(pt[0]), int(pt[1])
            z = None

            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                d = depth[y, x]
                if np.isfinite(d) and d > 1e-6 and (mask[y, x] if mask is not None else True):
                    z = float(d)

            if z is None and center_index is not None and idx == center_index:
                points_3d.append(body_pos.copy())
                continue

            if z is None and depths is not None and len(xs) > 0:
                r = max(0, int(self.depth_search_radius))
                if r > 0:
                    x0 = max(0, x - r)
                    x1 = min(depth.shape[1], x + r + 1)
                    y0 = max(0, y - r)
                    y1 = min(depth.shape[0], y + r + 1)
                    local_valid = valid[y0:y1, x0:x1]
                    if np.any(local_valid):
                        ly, lx = np.where(local_valid)
                        lx = lx + x0
                        ly = ly + y0
                        dx = lx - x
                        dy = ly - y
                        dist2 = dx * dx + dy * dy
                        nearest = int(np.argmin(dist2))
                        z = float(depth[ly[nearest], lx[nearest]])
                        x = int(lx[nearest])
                        y = int(ly[nearest])

                if z is None:
                    dx = xs - x
                    dy = ys - y
                    dist2 = dx * dx + dy * dy
                    nearest = int(np.argmin(dist2))
                    z = float(depths[nearest])
                    x = int(xs[nearest])
                    y = int(ys[nearest])

            if z is None:
                points_3d.append(body_pos.copy())
                continue

            x_cam = (x - intr.cx) / intr.fx * z
            y_cam = (intr.cy - y) / intr.fy * z
            world = cam_pos + cam_right * x_cam + cam_up * y_cam + cam_forward * z
            points_3d.append(world.astype(np.float64))

        return np.array(points_3d)
