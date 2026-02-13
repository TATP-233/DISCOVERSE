"""
3DGS to COLMAP converter - 围绕3DGS物体几何中心旋转拍摄并导出COLMAP格式数据

功能:
    1. 加载3DGS PLY文件，计算几何中心
    2. 生成围绕中心旋转的相机轨迹
    3. 渲染RGB图像
    4. 导出为COLMAP格式 (cameras.txt, images.txt, points3D.txt)

使用方法:
    python gs2colmap.py --gsply /path/to/model.ply --num-views 36 --output /path/to/output
"""

import os
import argparse
import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData
from dataclasses import dataclass
from typing import List, Tuple

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
from discoverse.utils import camera2k


def load_ply_center(ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载PLY文件并计算几何中心和边界

    Returns:
        center: 几何中心 (3,)
        min_bound: 最小边界 (3,)
        max_bound: 最大边界 (3,)
    """
    plydata = PlyData.read(ply_path)
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    center = xyz.mean(axis=0)
    min_bound = xyz.min(axis=0)
    max_bound = xyz.max(axis=0)

    return center, min_bound, max_bound


def load_ply_points(ply_path: str, max_points: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """
    从3DGS PLY文件中提取点云数据

    Args:
        ply_path: PLY文件路径
        max_points: 最大点数（避免点云过大）

    Returns:
        xyz: 点位置 (N, 3)
        rgb: 点颜色 (N, 3) uint8
    """
    plydata = PlyData.read(ply_path)

    # 提取位置
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    # 从SH系数的DC分量计算颜色
    # 3DGS中颜色存储为球谐系数，DC分量(f_dc_0, f_dc_1, f_dc_2)可以转换为RGB
    C0 = 0.28209479177387814  # SH基函数常数
    f_dc_0 = np.asarray(plydata.elements[0]["f_dc_0"])
    f_dc_1 = np.asarray(plydata.elements[0]["f_dc_1"])
    f_dc_2 = np.asarray(plydata.elements[0]["f_dc_2"])

    # SH DC to RGB: color = sh * C0 + 0.5
    r = np.clip((f_dc_0 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    g = np.clip((f_dc_1 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    b = np.clip((f_dc_2 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=1)

    # 如果点数过多，进行随机采样
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]

    return xyz, rgb


def generate_random_points(min_bound: np.ndarray, max_bound: np.ndarray,
                           num_points: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """
    在边界范围内生成随机点云

    Args:
        min_bound: 最小边界 (3,)
        max_bound: 最大边界 (3,)
        num_points: 生成的点数

    Returns:
        xyz: 点位置 (N, 3)
        rgb: 点颜色 (N, 3) uint8，随机灰度
    """
    # 在边界范围内均匀随机采样
    xyz = np.random.uniform(min_bound, max_bound, size=(num_points, 3))

    # 生成随机灰度颜色 (用于可视化)
    gray = np.random.randint(100, 200, size=(num_points, 1), dtype=np.uint8)
    rgb = np.repeat(gray, 3, axis=1)

    return xyz, rgb


def generate_orbit_trajectory(
    center: np.ndarray,
    radius: float,
    num_views: int,
    elevation_angles: List[float] = [0.0],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成围绕中心点旋转的相机轨迹

    使用与camera_view.py相同的坐标系约定：
    - yaw: 水平方向旋转角度 (绕Z轴)
    - pitch: 俯仰角度 (正值向下看)

    Args:
        center: 旋转中心点 (3,)
        radius: 旋转半径
        num_views: 每个仰角的视角数量
        elevation_angles: 仰角列表 (度)，正值表示相机在物体上方

    Returns:
        轨迹列表: [(position, quaternion_wxyz), ...]
    """
    trajectory = []

    for elev_deg in elevation_angles:
        elev_rad = np.deg2rad(elev_deg)

        for i in range(num_views):
            # 方位角 (绕Z轴旋转)
            azimuth = 2 * np.pi * i / num_views

            # 计算相机位置 (球坐标系)
            # elev_rad > 0 时相机在上方，elev_rad < 0 时相机在下方
            x = center[0] + radius * np.cos(elev_rad) * np.cos(azimuth)
            y = center[1] + radius * np.cos(elev_rad) * np.sin(azimuth)
            z = center[2] + radius * np.sin(elev_rad)
            position = np.array([x, y, z])

            # 计算相机朝向中心的旋转
            # camera_view.py中：yaw控制水平朝向，pitch控制俯仰
            # 相机朝向中心需要：yaw = azimuth + π（转180度面向中心）
            camera_yaw = azimuth + np.pi
            # pitch：相机在上方时需要向下看（pitch > 0），在下方时向上看（pitch < 0）
            camera_pitch = elev_rad

            # 使用欧拉角转四元数 (与camera_view.py一致: xyz顺序)
            quat_xyzw = Rotation.from_euler("xyz", [0.0, camera_pitch, camera_yaw]).as_quat()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

            trajectory.append((position, quat_wxyz))

    return trajectory


class GS2ColmapEnv(SimulatorBase):
    """用于3DGS渲染的简化环境类"""

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def updateControl(self, action):
        pass

    def post_load_mjcf(self):
        pass

    def getObservation(self):
        rgb_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_rgb_cam_id]
        self.obs = {
            "rgb_cam_posi": rgb_cam_pose_lst,
            "rgb_img": self.img_rgb_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def checkTerminated(self):
        return False

    def getReward(self):
        return None


def write_colmap_cameras(output_path: str, camera_id: int, width: int, height: int,
                         fx: float, fy: float, cx: float, cy: float):
    """
    写入COLMAP cameras.txt文件
    格式: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    使用PINHOLE模型: fx, fy, cx, cy
    """
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{camera_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_colmap_images(output_path: str, images_data: List[dict]):
    """
    写入COLMAP images.txt文件
    格式:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] (空行)

    注意: COLMAP使用world-to-camera变换，需要对位姿取逆
    """
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_data)}\n")

        for img_data in images_data:
            image_id = img_data['image_id']
            camera_id = img_data['camera_id']
            name = img_data['name']

            # 获取camera-to-world变换
            position = np.array(img_data['position'])  # 相机位置 (世界坐标)
            quat_wxyz = np.array(img_data['quaternion'])  # 相机旋转 (wxyz)

            # 转换为world-to-camera (COLMAP格式)
            # R_wc = R_cw^T, t_wc = -R_wc * t_cw
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            rot_cw = Rotation.from_quat(quat_xyzw)
            rot_wc = rot_cw.inv()

            # COLMAP的t是世界原点在相机坐标系下的位置
            t_wc = -rot_wc.as_matrix() @ position

            quat_wc_xyzw = rot_wc.as_quat()
            quat_wc_wxyz = quat_wc_xyzw[[3, 0, 1, 2]]

            f.write(f"{image_id} {quat_wc_wxyz[0]:.9f} {quat_wc_wxyz[1]:.9f} {quat_wc_wxyz[2]:.9f} {quat_wc_wxyz[3]:.9f} ")
            f.write(f"{t_wc[0]:.9f} {t_wc[1]:.9f} {t_wc[2]:.9f} {camera_id} {name}\n")
            f.write("\n")  # POINTS2D 空行


def write_colmap_points3d(output_path: str, xyz: np.ndarray = None, rgb: np.ndarray = None):
    """
    写入COLMAP points3D.txt文件

    Args:
        output_path: 输出文件路径
        xyz: 点位置 (N, 3)，如果为None则写入空文件
        rgb: 点颜色 (N, 3) uint8
    """
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        if xyz is None or len(xyz) == 0:
            f.write("# Number of points: 0\n")
        else:
            f.write(f"# Number of points: {len(xyz)}\n")
            for i in range(len(xyz)):
                x, y, z = xyz[i]
                r, g, b = rgb[i] if rgb is not None else (128, 128, 128)
                # POINT3D_ID X Y Z R G B ERROR TRACK[]
                # ERROR设为0，TRACK为空（没有2D点对应关系）
                f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0\n")


def main():
    parser = argparse.ArgumentParser(description='3DGS to COLMAP converter')
    parser.add_argument('--gsply', type=str, required=True,
                        help='3DGS PLY文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录路径，默认为PLY文件同目录下以PLY文件名命名的文件夹')
    parser.add_argument('--num-views', type=int, default=36,
                        help='每个仰角的视角数量 (默认: 36)')
    parser.add_argument('--elevation-angles', type=str, default='0,30,-30',
                        help='仰角列表，逗号分隔 (默认: 0,30,-30)')
    parser.add_argument('--radius-scale', type=float, default=1.2,
                        help='旋转半径相对于物体尺寸的比例 (默认: 1.2)')
    parser.add_argument('--radius', type=float, default=None,
                        help='固定旋转半径，设置后会忽略radius-scale')
    parser.add_argument('--fovy', type=float, default=60.0,
                        help='相机垂直视场角 (度，默认: 60)')
    parser.add_argument('--width', type=int, default=1280,
                        help='图像宽度 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='图像高度 (默认: 720)')
    parser.add_argument('--show-viewer', action='store_true',
                        help='显示渲染窗口')
    parser.add_argument('--max-points', type=int, default=50000,
                        help='points3D最大点数 (默认: 50000，设为0则不生成点云)')
    parser.add_argument('--random-points', action='store_true',
                        help='使用随机点云而非从3DGS提取点云')
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.gsply):
        raise FileNotFoundError(f"PLY文件不存在: {args.gsply}")

    # 设置输出目录：默认使用PLY文件名（不带扩展名）作为输出目录名
    if args.output is None:
        ply_basename = os.path.splitext(os.path.basename(args.gsply))[0]
        args.output = os.path.join(os.path.dirname(args.gsply), ply_basename)

    # 创建输出目录结构
    images_dir = os.path.join(args.output, "images")
    sparse_dir = os.path.join(args.output, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    print(f"加载PLY文件: {args.gsply}")
    center, min_bound, max_bound = load_ply_center(args.gsply)
    object_size = np.linalg.norm(max_bound - min_bound)

    print(f"物体中心: {center}")
    print(f"物体边界: {min_bound} ~ {max_bound}")
    print(f"物体尺寸: {object_size:.3f}")

    # 计算旋转半径
    if args.radius is not None:
        radius = args.radius
    else:
        radius = object_size * args.radius_scale
    print(f"旋转半径: {radius:.3f}")

    # 解析仰角
    elevation_angles = [float(x.strip()) for x in args.elevation_angles.split(',')]
    print(f"仰角列表: {elevation_angles}")

    # 生成轨迹
    trajectory = generate_orbit_trajectory(
        center=center,
        radius=radius,
        num_views=args.num_views,
        elevation_angles=elevation_angles,
    )
    total_views = len(trajectory)
    print(f"总视角数: {total_views}")

    # 计算相机内参
    fovy_rad = np.deg2rad(args.fovy)
    K = camera2k(fovy_rad, args.width, args.height)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 构建MuJoCo场景XML (只需要一个简单的相机)
    camera_env_xml = f"""
    <mujoco model="gs2colmap_env">
      <option integrator="RK4" solver="Newton" gravity="0 0 0"/>
      <worldbody>
        <body name="camera_body" pos="{trajectory[0][0][0]} {trajectory[0][0][1]} {trajectory[0][0][2]}"
              quat="{trajectory[0][1][0]} {trajectory[0][1][1]} {trajectory[0][1][2]} {trajectory[0][1][3]}">
          <camera name="main_camera" fovy="{args.fovy}" pos="0 0 0" quat="0.5 0.5 -0.5 -0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    # 临时保存XML
    xml_save_path = os.path.join(args.output, "_temp_env.xml")
    with open(xml_save_path, "w") as f:
        f.write(camera_env_xml)

    # 配置环境
    cfg = BaseConfig()
    cfg.gsply = args.gsply
    cfg.render_set["fps"] = 30
    cfg.render_set["width"] = args.width
    cfg.render_set["height"] = args.height
    cfg.timestep = 1.0 / cfg.render_set["fps"]
    cfg.decimation = 1
    cfg.mjcf_file_path = xml_save_path
    cfg.obs_rgb_cam_id = [0]
    cfg.obs_depth_cam_id = []
    cfg.sync = not args.show_viewer  # 非交互模式下不同步

    # 配置3DGS渲染器
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = args.gsply

    # 创建环境
    env = GS2ColmapEnv(cfg)

    # 清理临时XML
    os.remove(xml_save_path)

    # 渲染并保存图像
    images_data = []
    env.reset()

    print(f"\n开始渲染 {total_views} 个视角...")
    for i, (position, quat_wxyz) in enumerate(trajectory):
        # 更新相机位姿
        env.mj_model.body("camera_body").pos[:] = position
        env.mj_model.body("camera_body").quat[:] = quat_wxyz

        # 执行一步获取渲染结果
        obs, _, _, _, _ = env.step()

        # 保存图像
        rgb_img = obs["rgb_img"][0]
        image_name = f"{i:06d}.png"
        image_path = os.path.join(images_dir, image_name)
        cv2.imwrite(image_path, rgb_img)

        # 记录相机参数
        cam_pose = obs["rgb_cam_posi"][0]
        images_data.append({
            'image_id': i + 1,  # COLMAP从1开始
            'camera_id': 1,
            'name': image_name,
            'position': cam_pose[0].tolist(),
            'quaternion': cam_pose[1].tolist()
        })

        if (i + 1) % 10 == 0 or i == total_views - 1:
            print(f"  进度: {i + 1}/{total_views}")

    # 写入COLMAP文件
    print("\n写入COLMAP格式文件...")
    write_colmap_cameras(
        os.path.join(sparse_dir, "cameras.txt"),
        camera_id=1,
        width=args.width,
        height=args.height,
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    write_colmap_images(
        os.path.join(sparse_dir, "images.txt"),
        images_data
    )

    # 生成点云
    if args.max_points > 0:
        if args.random_points:
            print(f"生成随机点云 ({args.max_points}点)...")
            points_xyz, points_rgb = generate_random_points(min_bound, max_bound, num_points=args.max_points)
        else:
            print(f"从3DGS提取点云 (最大{args.max_points}点)...")
            points_xyz, points_rgb = load_ply_points(args.gsply, max_points=args.max_points)
        print(f"  生成了 {len(points_xyz)} 个点")
        write_colmap_points3d(
            os.path.join(sparse_dir, "points3D.txt"),
            xyz=points_xyz,
            rgb=points_rgb
        )
    else:
        write_colmap_points3d(os.path.join(sparse_dir, "points3D.txt"))

    # 保存JSON格式的相机参数
    json_path = os.path.join(args.output, "camera_params.json")
    json_data = {
        'intrinsics': {
            'width': args.width,
            'height': args.height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'fovy': args.fovy
        },
        'images': images_data
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n完成! 输出目录: {args.output}")
    print(f"  - 图像: {images_dir} ({total_views}张)")
    print(f"  - COLMAP文件: {sparse_dir}")
    print(f"    - cameras.txt")
    print(f"    - images.txt")
    if args.max_points > 0:
        print(f"    - points3D.txt ({len(points_xyz)}点)")
    else:
        print(f"    - points3D.txt (空)")
    print(f"  - 相机参数: {json_path}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=500)
    main()
