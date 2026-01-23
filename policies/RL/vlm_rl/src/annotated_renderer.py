"""
AnnotatedRenderer: Render MuJoCo scene with keypoint annotations for VLM.

Renders RGB images from MuJoCo and overlays numbered keypoint markers,
similar to ReKep's visualization for VLM queries.
"""

import numpy as np
import mujoco
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class AnnotatedRenderer:
    """
    Render MuJoCo scene with numbered keypoint annotations.

    Creates images suitable for VLM consumption with clearly marked
    keypoint indices overlaid on the scene.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
        camera_name: Optional[str] = None,
    ):
        """
        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data
            width: Image width in pixels
            height: Image height in pixels
            camera_name: Name of camera to use (None for default free camera)
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.width = width
        self.height = height
        self.camera_name = camera_name

        # Initialize MuJoCo renderer
        self.renderer = mujoco.Renderer(mj_model, height, width)

        # Get camera ID if specified
        self.camera_id = -1
        if camera_name is not None:
            self.camera_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if self.camera_id < 0:
                print(f"Warning: Camera '{camera_name}' not found, using free camera")

    def render_rgb(self) -> np.ndarray:
        """
        Render RGB image from the scene.

        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        self.renderer.update_scene(self.mj_data, camera=self.camera_id)
        return self.renderer.render()

    def render_depth(self) -> np.ndarray:
        """
        Render depth image from the scene.

        Returns:
            Depth image as (H, W) float32 array (meters)
        """
        self.renderer.update_scene(self.mj_data, camera=self.camera_id)
        self.renderer.enable_depth_rendering(True)
        depth = self.renderer.render()
        self.renderer.enable_depth_rendering(False)
        return depth

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """
        Get camera intrinsic parameters.

        Returns:
            CameraIntrinsics object
        """
        if self.camera_id >= 0:
            fovy = self.mj_model.cam_fovy[self.camera_id]
        else:
            fovy = self.mj_model.vis.global_.fovy

        # Convert fovy to focal length
        fy = self.height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy  # Assuming square pixels

        return CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=self.width / 2.0,
            cy=self.height / 2.0,
            width=self.width,
            height=self.height,
        )

    def get_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera extrinsic parameters (position and rotation matrix).

        Returns:
            position: (3,) camera position in world frame
            rotation: (3, 3) rotation matrix (camera to world)
        """
        if self.camera_id >= 0:
            pos = self.mj_data.cam_xpos[self.camera_id].copy()
            rot = self.mj_data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        else:
            # Free camera - would need to track this separately
            pos = np.array([0.0, -1.5, 1.5])  # Default view
            rot = np.eye(3)

        return pos, rot

    def project_points_to_image(
        self,
        points_3d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: (N, 3) points in world coordinates

        Returns:
            points_2d: (N, 2) pixel coordinates
            valid_mask: (N,) boolean mask for points in front of camera
        """
        intrinsics = self.get_camera_intrinsics()
        cam_pos, cam_rot = self.get_camera_extrinsics()

        # Transform to camera frame
        # MuJoCo camera looks along -z in camera frame
        points_cam = (points_3d - cam_pos) @ cam_rot

        # Check if points are in front of camera
        # In MuJoCo camera frame, z points backward, so valid points have negative z
        valid_mask = points_cam[:, 2] < 0

        # Project to image plane
        # Flip z because camera looks along -z
        z = -points_cam[:, 2]
        z = np.where(z > 0, z, 1e-6)  # Avoid division by zero

        u = intrinsics.fx * points_cam[:, 0] / z + intrinsics.cx
        v = intrinsics.fy * (-points_cam[:, 1]) / z + intrinsics.cy  # Flip y

        points_2d = np.stack([u, v], axis=1)

        # Also check if points are within image bounds
        valid_mask = valid_mask & (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.height)
        )

        return points_2d, valid_mask

    def render_with_keypoints(
        self,
        keypoints_3d: np.ndarray,
        marker_radius: int = 8,
        font_scale: float = 0.5,
        font_thickness: int = 2,
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Render scene with numbered keypoint markers.

        Args:
            keypoints_3d: (N, 3) keypoint world coordinates
            marker_radius: Radius of keypoint markers in pixels
            font_scale: Font scale for keypoint numbers
            font_thickness: Font thickness
            colors: Optional list of (B, G, R) colors for each keypoint.
                    If None, uses a default color scheme.

        Returns:
            Annotated RGB image as (H, W, 3) uint8 array
        """
        # Render base image
        rgb = self.render_rgb().copy()

        if len(keypoints_3d) == 0:
            return rgb

        # Project keypoints to 2D
        points_2d, valid_mask = self.project_points_to_image(keypoints_3d)

        # Default colors: cycle through a palette
        if colors is None:
            palette = [
                (255, 100, 100),   # Red
                (100, 255, 100),   # Green
                (100, 100, 255),   # Blue
                (255, 255, 100),   # Yellow
                (255, 100, 255),   # Magenta
                (100, 255, 255),   # Cyan
                (255, 165, 0),     # Orange
                (147, 112, 219),   # Purple
            ]
            colors = [palette[i % len(palette)] for i in range(len(keypoints_3d))]

        # Draw keypoints
        for i, (pt_2d, valid) in enumerate(zip(points_2d, valid_mask)):
            if not valid:
                continue

            x, y = int(pt_2d[0]), int(pt_2d[1])
            color = colors[i]

            # Draw filled circle with white border
            cv2.circle(rgb, (x, y), marker_radius + 2, (255, 255, 255), -1)
            cv2.circle(rgb, (x, y), marker_radius, color, -1)

            # Draw keypoint index
            text = str(i)
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )[0]

            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2

            # White background for text readability
            cv2.putText(
                rgb, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), font_thickness + 2
            )
            # Black text
            cv2.putText(
                rgb, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), font_thickness
            )

        return rgb

    def render_with_keypoints_by_object(
        self,
        keypoints_3d: np.ndarray,
        object_keypoint_ranges: dict,
        marker_radius: int = 8,
        font_scale: float = 0.5,
        font_thickness: int = 2,
    ) -> np.ndarray:
        """
        Render with keypoints colored by object.

        Args:
            keypoints_3d: (N, 3) keypoint world coordinates
            object_keypoint_ranges: {object_name: (start_idx, end_idx)}
            marker_radius: Radius of keypoint markers
            font_scale: Font scale for numbers
            font_thickness: Font thickness

        Returns:
            Annotated RGB image
        """
        # Assign colors by object
        palette = [
            (255, 100, 100),   # Red
            (100, 255, 100),   # Green
            (100, 100, 255),   # Blue
            (255, 255, 100),   # Yellow
            (255, 100, 255),   # Magenta
            (100, 255, 255),   # Cyan
        ]

        colors = [(128, 128, 128)] * len(keypoints_3d)  # Default gray

        for obj_idx, (obj_name, (start, end)) in enumerate(object_keypoint_ranges.items()):
            obj_color = palette[obj_idx % len(palette)]
            for i in range(start, end):
                if i < len(colors):
                    colors[i] = obj_color

        return self.render_with_keypoints(
            keypoints_3d,
            marker_radius=marker_radius,
            font_scale=font_scale,
            font_thickness=font_thickness,
            colors=colors,
        )

    def save_annotated_image(
        self,
        filepath: str,
        keypoints_3d: np.ndarray,
        **kwargs
    ) -> None:
        """
        Render and save annotated image to file.

        Args:
            filepath: Output file path
            keypoints_3d: (N, 3) keypoint world coordinates
            **kwargs: Additional arguments for render_with_keypoints
        """
        img = self.render_with_keypoints(keypoints_3d, **kwargs)
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def generate_keypoint_description(
        self,
        keypoints_3d: np.ndarray,
        object_keypoint_ranges: dict,
    ) -> str:
        """
        Generate text description of keypoints for VLM prompt.

        Args:
            keypoints_3d: (N, 3) keypoint coordinates
            object_keypoint_ranges: {object_name: (start_idx, end_idx)}

        Returns:
            Text description of keypoints
        """
        desc = "Keypoints in the scene:\n"

        for obj_name, (start, end) in object_keypoint_ranges.items():
            desc += f"\n{obj_name}:\n"
            for i in range(start, end):
                if i < len(keypoints_3d):
                    pos = keypoints_3d[i]
                    desc += f"  - Keypoint {i}: position [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"

        desc += "\nNote: Keypoints are numbered 0 to N-1. Use keypoints[i] to reference keypoint i in constraints.\n"

        return desc
