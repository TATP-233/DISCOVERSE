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

        # Free camera settings (can be overridden after initialization)
        self.free_camera_lookat = [0.4, 0.0, 0.45]
        self.free_camera_distance = 1.2
        self.free_camera_azimuth = 180
        self.free_camera_elevation = -30

    def _build_mjv_camera(self) -> mujoco.MjvCamera:
        """Create a MuJoCo camera configured to match the renderer settings."""
        camera = mujoco.MjvCamera()
        if self.camera_id >= 0:
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = self.camera_id
        else:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Start from MuJoCo defaults to keep internal fields consistent.
            mujoco.mjv_defaultFreeCamera(self.mj_model, camera)
            camera.lookat = np.array(self.free_camera_lookat)
            camera.distance = self.free_camera_distance
            camera.azimuth = self.free_camera_azimuth
            camera.elevation = self.free_camera_elevation
        return camera

    def _get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get camera position and basis vectors in model/world coordinates."""
        camera = self._build_mjv_camera()
        pos = np.zeros(3, dtype=np.float64)
        forward = np.zeros(3, dtype=np.float64)
        up = np.zeros(3, dtype=np.float64)
        right = np.zeros(3, dtype=np.float64)
        mujoco.mjv_cameraFrame(pos, forward, up, right, self.mj_data, camera)
        return pos, forward, up, right

    def get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Public accessor for camera frame (position, forward, up, right)."""
        return self._get_camera_frame()

    def render_rgb(self) -> np.ndarray:
        """
        Render RGB image from the scene.

        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        camera = self._build_mjv_camera()
        self.renderer.update_scene(self.mj_data, camera=camera)
        return self.renderer.render()

    def render_depth(self) -> np.ndarray:
        """
        Render depth image from the scene.

        Returns:
            Depth image as (H, W) float32 array (meters)
        """
        camera = self._build_mjv_camera()
        self.renderer.update_scene(self.mj_data, camera=camera)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth

    def render_segmentation(self) -> np.ndarray:
        """
        Render segmentation image from the scene.

        Returns:
            Segmentation image as (H, W, 2) int32 array:
            [:, :, 0] = object id, [:, :, 1] = object type.
        """
        camera = self._build_mjv_camera()
        self.renderer.update_scene(self.mj_data, camera=camera)
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return seg

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        """
        Get camera intrinsic parameters.

        Returns:
            CameraIntrinsics object
        """
        if self.camera_id >= 0:
            fovy = self.mj_model.cam_fovy[self.camera_id]
            if fovy <= 1e-6:
                fovy = self.mj_model.vis.global_.fovy
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

    def render_with_keypoints_2d(
        self,
        keypoints_2d: np.ndarray,
        marker_radius: int = 8,
        font_scale: float = 0.5,
        font_thickness: int = 2,
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Render scene with numbered 2D keypoint markers.

        Args:
            keypoints_2d: (N, 2) keypoint pixel coordinates
            marker_radius: Radius of keypoint markers in pixels
            font_scale: Font scale for keypoint numbers
            font_thickness: Font thickness
            colors: Optional list of (B, G, R) colors for each keypoint

        Returns:
            Annotated RGB image as (H, W, 3) uint8 array
        """
        rgb = self.render_rgb().copy()

        if keypoints_2d is None or len(keypoints_2d) == 0:
            return rgb

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
            colors = [palette[i % len(palette)] for i in range(len(keypoints_2d))]

        for i, pt_2d in enumerate(keypoints_2d):
            x, y = int(round(pt_2d[0])), int(round(pt_2d[1]))
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            color = colors[i]
            cv2.circle(rgb, (x, y), marker_radius + 2, (255, 255, 255), -1)
            cv2.circle(rgb, (x, y), marker_radius, color, -1)

            text = str(i)
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )[0]

            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2

            cv2.putText(
                rgb, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), font_thickness + 2
            )
            cv2.putText(
                rgb, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), font_thickness
            )

        return rgb

    def render_with_keypoints_2d_by_object(
        self,
        keypoints_2d: np.ndarray,
        object_keypoint_ranges: dict,
        marker_radius: int = 8,
        font_scale: float = 0.5,
        font_thickness: int = 2,
    ) -> np.ndarray:
        """
        Render with 2D keypoints colored by object.

        Args:
            keypoints_2d: (N, 2) keypoint pixel coordinates
            object_keypoint_ranges: {object_name: (start_idx, end_idx)}
            marker_radius: Radius of keypoint markers
            font_scale: Font scale for numbers
            font_thickness: Font thickness

        Returns:
            Annotated RGB image
        """
        palette = [
            (255, 100, 100),   # Red
            (100, 255, 100),   # Green
            (100, 100, 255),   # Blue
            (255, 255, 100),   # Yellow
            (255, 100, 255),   # Magenta
            (100, 255, 255),   # Cyan
        ]

        colors = [(128, 128, 128)] * len(keypoints_2d)

        for obj_idx, (obj_name, (start, end)) in enumerate(object_keypoint_ranges.items()):
            obj_color = palette[obj_idx % len(palette)]
            for i in range(start, end):
                if i < len(colors):
                    colors[i] = obj_color

        return self.render_with_keypoints_2d(
            keypoints_2d,
            marker_radius=marker_radius,
            font_scale=font_scale,
            font_thickness=font_thickness,
            colors=colors,
        )

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
