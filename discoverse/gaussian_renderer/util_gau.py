import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
import torch
import glm


def multiple_quaternion_vector3d(qwxyz, vxyz):
    qw = qwxyz[..., 0]
    qx = qwxyz[..., 1]
    qy = qwxyz[..., 2]
    qz = qwxyz[..., 3]
    vx = vxyz[..., 0]
    vy = vxyz[..., 1]
    vz = vxyz[..., 2]
    qvw = -vx * qx - vy * qy - vz * qz
    qvx = vx * qw - vy * qz + vz * qy
    qvy = vx * qz + vy * qw - vz * qx
    qvz = -vx * qy + vy * qx + vz * qw
    vx_ = qvx * qw - qvw * qx + qvz * qy - qvy * qz
    vy_ = qvy * qw - qvz * qx - qvw * qy + qvx * qz
    vz_ = qvz * qw + qvy * qx - qvx * qy - qvw * qz
    return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)


def multiple_quaternions(qwxyz1, qwxyz2):
    q1w = qwxyz1[..., 0]
    q1x = qwxyz1[..., 1]
    q1y = qwxyz1[..., 2]
    q1z = qwxyz1[..., 3]

    q2w = qwxyz2[..., 0]
    q2x = qwxyz2[..., 1]
    q2y = qwxyz2[..., 2]
    q2z = qwxyz2[..., 3]

    qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
    qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
    qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

    return torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)


class Camera:
    def __init__(self, h, w):
        self.znear = 1e-6
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = 1.05  # 0.8955465 # 0.6545
        self.position = np.array([0.0, 0.0, -2.0]).astype(np.float32)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, 1.0, 0.0]).astype(np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True

        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True

        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False

        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.0

    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_project_matrix(self):
        project_mat = glm.perspective(self.fovy, self.w / self.h, self.znear, self.zfar)
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_leftmouse_pressed:
            self.yaw += xoffset * self.rot_sensitivity
            self.pitch += yoffset * self.rot_sensitivity

            self.pitch = np.clip(self.pitch, -np.pi / 2, np.pi / 2)

            front = np.array(
                [
                    np.cos(self.yaw) * np.cos(self.pitch),
                    np.sin(self.pitch),
                    np.sin(self.yaw) * np.cos(self.pitch),
                ]
            )
            front = self._global_rot_mat() @ front.reshape(3, 1)
            front = front[:, 0]
            self.position[:] = (
                -front * np.linalg.norm(self.position - self.target) + self.target
            )

            self.is_pose_dirty = True

        if self.is_rightmouse_pressed:
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            right = np.cross(self.up, front)
            self.position += right * xoffset * self.trans_sensitivity
            self.target += right * xoffset * self.trans_sensitivity
            cam_up = np.cross(right, front)
            self.position += cam_up * yoffset * self.trans_sensitivity
            self.target += cam_up * yoffset * self.trans_sensitivity

            self.is_pose_dirty = True

    def process_wheel(self, dx, dy):
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        self.position += front * dy * self.zoom_sensitivity
        self.target += front * dy * self.zoom_sensitivity
        self.is_pose_dirty = True

    def process_roll_key(self, d):
        front = self.target - self.position
        right = np.cross(front, self.up)
        new_up = self.up + right * (d * self.roll_sensitivity / np.linalg.norm(right))
        self.up = new_up / np.linalg.norm(new_up)
        self.is_pose_dirty = True

    def flip_ground(self):
        self.up = -self.up
        self.is_pose_dirty = True

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist

    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True


@dataclass
class GaussianData:
    def __init__(self, xyz, rot, scale, opacity, sh):
        self.xyz = xyz
        self.rot = rot
        self.scale = scale
        self.opacity = opacity
        self.sh = sh

        self.origin_xyz = np.zeros(3)
        self.origin_rot = np.array([1.0, 0.0, 0.0, 0.0])

    def flat(self) -> np.ndarray:
        ret = np.concatenate(
            [self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1
        )
        return np.ascontiguousarray(ret)

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        return self.sh.shape[-1]


def gamma_shs(shs, gamma):
    C0 = 0.28209479177387814
    new_shs = ((np.clip(shs * C0 + 0.5, 0.0, 1.0) ** gamma) - 0.5) / C0
    return new_shs


def load_ply(path, gamma=1):
    max_sh_degree = 0
    plydata = PlyData.read(path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

    # assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, len(extra_f_names) // 3)
    )
    features_extra = features_extra[:, :, : (max_sh_degree + 1) ** 2 - 1]
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))
    opacities = opacities.astype(np.float32)

    if abs(gamma - 1.0) > 1e-3:
        features_dc = gamma_shs(features_dc, gamma)
        features_extra[..., :] = 0.0
        opacities *= 0.8

    shs = np.concatenate(
        [features_dc.reshape(-1, 3), features_extra.reshape(len(features_dc), -1)],
        axis=-1,
    ).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)
