"""
场景随机化实现

提供统一的场景随机化接口，支持物体位置、姿态和相机视角的随机化。
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing import Dict, List, Any, Optional, Tuple
from discoverse.utils import get_random_texture

class SceneRandomizer:
    """场景随机化器"""
    
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        初始化场景随机化器
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.viewer = None

        # 存储初始状态
        self.initial_camera_poses = {}
        
        # 保存初始相机姿态
        for cam_id in range(mj_model.ncam):
            cam_name = mj_model.camera(cam_id).name
            if cam_name:
                self.initial_camera_poses[cam_name] = {
                    'pos': mj_model.camera(cam_id).pos.copy(),
                    'quat': mj_model.camera(cam_id).quat.copy()
                }
        
        # 保存初始光照状态
        self.initial_light_states = {}
        if mj_model.nlight > 0:
            self.initial_light_states = {
                'pos': mj_model.light_pos.copy(),
                'dir': mj_model.light_dir.copy(),
                'ambient': mj_model.light_ambient.copy(),
                'diffuse': mj_model.light_diffuse.copy(),
                'specular': mj_model.light_specular.copy(),
                'active': mj_model.light_active.copy(),
            }

        self.free_body_qpos_ids = {}
        for i in range(self.mj_model.nbody):
            if len(self.mj_model.body(i).name) and self.mj_model.body(i).dofnum == 6:
                jq_id = np.where(self.mj_model.jnt_bodyid == self.mj_model.body(i).id)[0]
                if jq_id.size:
                    self.free_body_qpos_ids[self.mj_model.body(i).name] = int(jq_id[0])

    def set_viewer(self, viewer):
        """设置可视化器引用"""
        self.viewer = viewer

    def exec_randomization(self, randomization_config: Dict[str, Any], max_attempts: int = 100) -> bool:
        """
        根据配置随机化场景
        
        Args:
            randomization_config: 随机化配置
            max_attempts: 最大尝试次数（避免位置冲突时的无限循环）
            
        Returns:
            是否成功随机化
        """
        # 随机化物体 - 检查激活状态
        if 'objects' in randomization_config:
            objects_config = randomization_config['objects']
            # 如果objects是一个字典且包含activate字段
            if isinstance(objects_config, dict) and 'activate' in objects_config:
                if objects_config.get('activate', True):  # 默认激活
                    # 如果有objects列表，则使用它；否则跳过
                    if 'objects' in objects_config:
                        self._randomize_objects(objects_config['objects'], max_attempts)
                    else:
                        print("⚠️ objects配置中未找到具体物体列表")
                else:
                    print("📋 物体随机化已禁用")
            # 如果objects是一个列表（旧格式）
            elif isinstance(objects_config, list):
                self._randomize_objects(objects_config, max_attempts)
            else:
                print("⚠️ 无效的objects配置格式")
        
        # 随机化相机 - 检查激活状态
        if 'cameras' in randomization_config:
            cameras_config = randomization_config['cameras']
            if cameras_config.get('activate', True):  # 默认激活
                # 移除activate字段传递给具体的随机化方法
                cameras_config_clean = {k: v for k, v in cameras_config.items() if k != 'activate'}
                self._randomize_cameras(cameras_config_clean)
        
        # 随机化光照 - 检查激活状态
        if 'lighting' in randomization_config:
            lighting_config = randomization_config['lighting']
            if lighting_config.get('activate', True):  # 默认激活
                self._randomize_lighting(lighting_config)
        
        # 随机化桌面高度 - 检查激活状态
        if 'table_height' in randomization_config:
            table_config = randomization_config['table_height']
            if table_config.get('activate', True):  # 默认激活
                self._randomize_table_height(table_config)
        
        # 随机化材质 - 新功能
        if 'textures' in randomization_config:
            textures_config = randomization_config['textures']
            if textures_config.get('activate', True):  # 默认激活
                self._randomize_textures(textures_config)
                self.viewer
        
        # 应用更改
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        return True
    
    def _object_pose(self, body_name):
        """获取物体的位姿（位置xyz和朝向wxyz）"""
        try:
            qid = self.mj_model.jnt_qposadr[self.free_body_qpos_ids[body_name]]
            return self.mj_data.qpos[qid:qid+7][...]
        except KeyError:
            raise KeyError(f"Body name '{body_name}' not found in free_body_qpos_ids. Available bodies: {list(self.free_body_qpos_ids.keys())}")

    def _randomize_objects(self, objects_config: List[Dict[str, Any]], max_attempts: int) -> bool:
        """
        随机化物体位置和姿态
        
        Args:
            objects_config: 物体随机化配置列表
            max_attempts: 最大尝试次数
            
        Returns:
            是否成功
        """
        return self._randomize_objects_simple(objects_config)
    
    def _randomize_objects_simple(self, objects_config: List[Dict[str, Any]]) -> bool:
        """
        简单物体随机化（不考虑碰撞检测）
        
        Args:
            objects_config: 物体随机化配置列表
            
        Returns:
            是否成功
        """
        for obj_config in objects_config:
            success = self._randomize_single_object(obj_config)
            if not success:
                print(f"⚠️ 物体 '{obj_config.get('name', 'unknown')}' 随机化失败")
                return False
        
        return True
    
    def _randomize_single_object(self, obj_config: Dict[str, Any]) -> bool:
        """
        随机化单个物体
        
        Args:
            obj_config: 物体随机化配置
            
        Returns:
            是否成功
        """
        object_name = obj_config.get('name')
        if not object_name:
            return False

        try:
            joint_adr = self.mj_model.jnt_qposadr[self.free_body_qpos_ids[object_name]]
        except KeyError:
            print(f"❌ 未找到物体: {object_name} or {object_name} 没有free_joint")
            return False        
        
        # 随机化位置
        if 'position' in obj_config:
            self._randomize_object_position(joint_adr, obj_config['position'])
        
        # 随机化姿态
        if 'orientation' in obj_config:
            self._randomize_object_orientation(joint_adr, obj_config['orientation'])
        
        return True
    
    def _randomize_object_position(self, joint_adr: int, position_config: Dict[str, Any]):
        """
        随机化物体位置
        
        Args:
            joint_adr: 关节地址
            position_config: 位置随机化配置
        """
        # 获取当前位置
        current_pos = self.mj_data.qpos[joint_adr:joint_adr+3].copy()
        
        # 应用随机偏移
        if 'offset_range' in position_config:
            offset_range = position_config['offset_range']
            if isinstance(offset_range, (list, tuple)) and len(offset_range) == 3:
                # 每个轴独立的偏移范围 [x_range, y_range, z_range]
                offset = np.array([
                    2 * (np.random.random() - 0.5) * offset_range[0],
                    2 * (np.random.random() - 0.5) * offset_range[1],
                    2 * (np.random.random() - 0.5) * offset_range[2]
                ])
            elif isinstance(offset_range, (int, float)):
                # 统一的偏移范围
                offset = 2 * (np.random.random(3) - 0.5) * offset_range
            else:
                offset = np.zeros(3)
            
            self.mj_data.qpos[joint_adr:joint_adr+3] = current_pos + offset
        
        # 应用固定范围约束
        if 'bounds' in position_config:
            bounds = position_config['bounds']
            for i, (min_val, max_val) in enumerate(bounds):
                if i < 3:  # x, y, z
                    self.mj_data.qpos[joint_adr + i] = np.clip(
                        self.mj_data.qpos[joint_adr + i], min_val, max_val
                    )
    
    def _randomize_object_orientation(self, joint_adr: int, orientation_config: Dict[str, Any]):
        """
        随机化物体姿态
        
        Args:
            joint_adr: 关节地址
            orientation_config: 姿态随机化配置
        """
        # 获取当前四元数 (w, x, y, z)
        current_quat = self.mj_data.qpos[joint_adr+3:joint_adr+7].copy()
        
        if 'euler_range' in orientation_config:
            # 欧拉角随机化
            euler_range = orientation_config['euler_range']
            
            # 将当前四元数转换为欧拉角
            current_rotation = Rotation.from_quat(current_quat[[1, 2, 3, 0]])  # (x, y, z, w)
            current_euler = current_rotation.as_euler('xyz', degrees=False)
            
            # 应用随机偏移
            if isinstance(euler_range, (list, tuple)) and len(euler_range) == 3:
                euler_offset = np.array([
                    2 * (np.random.random() - 0.5) * euler_range[0],
                    2 * (np.random.random() - 0.5) * euler_range[1],
                    2 * (np.random.random() - 0.5) * euler_range[2]
                ])
            elif isinstance(euler_range, (int, float)):
                euler_offset = 2 * (np.random.random(3) - 0.5) * euler_range
            else:
                euler_offset = np.zeros(3)
            
            # 计算新的欧拉角并转换回四元数
            new_euler = current_euler + euler_offset
            new_rotation = Rotation.from_euler('xyz', new_euler, degrees=False)
            new_quat = new_rotation.as_quat()  # (x, y, z, w)
            
            # 转换为MuJoCo格式 (w, x, y, z)
            self.mj_data.qpos[joint_adr+3:joint_adr+7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
        
        elif 'random_rotation' in orientation_config and orientation_config['random_rotation']:
            # 完全随机旋转
            random_quat = self._generate_random_quaternion()
            self.mj_data.qpos[joint_adr+3:joint_adr+7] = random_quat
    
    def _randomize_cameras(self, cameras_config: Dict[str, Any]):
        """
        随机化相机视角
        
        Args:
            cameras_config: 相机随机化配置
        """
        for camera_name, camera_config in cameras_config.items():
            self._randomize_single_camera(camera_name, camera_config)
    
    def _randomize_single_camera(self, camera_name: str, camera_config: Dict[str, Any]):
        """
        随机化单个相机
        
        Args:
            camera_name: 相机名称
            camera_config: 相机随机化配置
        """
        try:
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                print(f"❌ 未找到相机: {camera_name}")
                return
            
            camera = self.mj_model.camera(cam_id)
            
            # 获取初始姿态
            initial_pose = self.initial_camera_poses.get(camera_name)
            if not initial_pose:
                return
            
            # 随机化位置
            if 'position_offset' in camera_config:
                offset_range = camera_config['position_offset']
                if isinstance(offset_range, (list, tuple)) and len(offset_range) == 3:
                    offset = np.array([
                        2 * (np.random.random() - 0.5) * offset_range[0],
                        2 * (np.random.random() - 0.5) * offset_range[1],
                        2 * (np.random.random() - 0.5) * offset_range[2]
                    ])
                elif isinstance(offset_range, (int, float)):
                    offset = 2 * (np.random.random(3) - 0.5) * offset_range
                else:
                    offset = np.zeros(3)
                
                camera.pos[:] = initial_pose['pos'] + offset
            
            # 随机化朝向
            if 'orientation_offset' in camera_config:
                euler_range = camera_config['orientation_offset']
                
                # 将当前四元数转换为欧拉角
                current_quat = initial_pose['quat'][[1, 2, 3, 0]]  # (x, y, z, w)
                current_rotation = Rotation.from_quat(current_quat)
                current_euler = current_rotation.as_euler('xyz', degrees=False)
                
                # 应用随机偏移
                if isinstance(euler_range, (list, tuple)) and len(euler_range) == 3:
                    euler_offset = np.array([
                        2 * (np.random.random() - 0.5) * euler_range[0],
                        2 * (np.random.random() - 0.5) * euler_range[1],
                        2 * (np.random.random() - 0.5) * euler_range[2]
                    ])
                elif isinstance(euler_range, (int, float)):
                    euler_offset = 2 * (np.random.random(3) - 0.5) * euler_range
                else:
                    euler_offset = np.zeros(3)
                
                # 计算新的欧拉角并转换回四元数
                new_euler = current_euler + euler_offset
                new_rotation = Rotation.from_euler('xyz', new_euler, degrees=False)
                new_quat = new_rotation.as_quat()  # (x, y, z, w)
                
                # 转换为MuJoCo格式 (w, x, y, z)
                camera.quat[:] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
            
        except Exception as e:
            print(f"❌ 随机化相机 '{camera_name}' 时出错: {e}")
    
    def _generate_random_quaternion(self) -> np.ndarray:
        """
        生成随机单位四元数
        
        Returns:
            随机四元数 (w, x, y, z)
        """
        # 使用均匀分布生成随机四元数
        u1, u2, u3 = np.random.random(3)
        
        sqrt1_u1 = np.sqrt(1 - u1)
        sqrt_u1 = np.sqrt(u1)
        
        w = sqrt1_u1 * np.sin(2 * np.pi * u2)
        x = sqrt1_u1 * np.cos(2 * np.pi * u2)
        y = sqrt_u1 * np.sin(2 * np.pi * u3)
        z = sqrt_u1 * np.cos(2 * np.pi * u3)
        
        return np.array([w, x, y, z])
    
    def _randomize_lighting(self, lighting_config: Dict[str, Any]):
        """
        随机化光照设置
        
        Args:
            lighting_config: 光照随机化配置
        """

        if self.mj_model.nlight == 0:
            return
        
        # 随机化光源颜色
        if lighting_config.get('random_color', False):
            if lighting_config.get('individual_colors', False):
                # 为每个光源单独设置颜色
                for i in range(self.mj_model.nlight):
                    self.mj_model.light_ambient[i, :] = np.random.random(3) * 0.1
                    self.mj_model.light_diffuse[i, :] = np.random.random(3)
                    self.mj_model.light_specular[i, :] = np.random.random(3) * 0.3
            else:
                # 所有光源使用相同的随机化
                self.mj_model.light_ambient[:] = np.random.random(size=self.mj_model.light_ambient.shape) * 0.1
                self.mj_model.light_diffuse[:] = np.random.random(size=self.mj_model.light_diffuse.shape)
                self.mj_model.light_specular[:] = np.random.random(size=self.mj_model.light_specular.shape) * 0.3
        
        # 随机化光源激活状态
        if lighting_config.get('random_active', False):
            active_prob = lighting_config.get('active_probability', 0.5)
            self.mj_model.light_active[:] = np.int32(np.random.rand(self.mj_model.nlight) > (1 - active_prob)).tolist()
            
            # 确保至少有一个光源是激活的
            if np.sum(self.mj_model.light_active) == 0:
                self.mj_model.light_active[np.random.randint(self.mj_model.nlight)] = 1
        
        # 随机化光源位置
        if 'position_offset' in lighting_config:
            pos_config = lighting_config['position_offset']
            
            # 应用位置偏移
            if isinstance(pos_config, dict):
                xy_scale = pos_config.get('xy_scale', 0.3)
                z_scale = pos_config.get('z_scale', 0.2)
                
                self.mj_model.light_pos[:, :2] = (
                    self.mj_model.light_pos0[:, :2] + 
                    np.random.normal(scale=xy_scale, size=self.mj_model.light_pos[:, :2].shape)
                )
                self.mj_model.light_pos[:, 2] = (
                    self.mj_model.light_pos0[:, 2] + 
                    np.random.normal(scale=z_scale, size=self.mj_model.light_pos[:, 2].shape)
                )
            else:
                print(f"unsupported position_offset format: {pos_config}")

        # 随机化光源方向
        if lighting_config.get('random_direction', False):
            # 生成随机方向向量
            self.mj_model.light_dir[:] = np.random.random(size=self.mj_model.light_dir.shape) - 0.5
            self.mj_model.light_dir[:, 2] *= 2.0  # Z方向偏向向下
            
            # 归一化方向向量
            norms = np.linalg.norm(self.mj_model.light_dir, axis=1, keepdims=True)
            self.mj_model.light_dir[:] = self.mj_model.light_dir / (norms + 1e-8)
            
            # 确保光源向下照射
            self.mj_model.light_dir[:, 2] = -np.abs(self.mj_model.light_dir[:, 2])
            
        # 随机化光强
        if 'intensity_range' in lighting_config:
            intensity_config = lighting_config['intensity_range']
            
            if isinstance(intensity_config, dict):
                min_intensity = intensity_config.get('min', 0.5)
                max_intensity = intensity_config.get('max', 1.0)
            else:
                min_intensity, max_intensity = 0.5, 1.0
            
            # 为每个颜色通道应用强度缩放
            intensity_scale = np.random.uniform(min_intensity, max_intensity, size=(self.mj_model.nlight, 1))
            
            self.mj_model.light_ambient[:] *= intensity_scale
            self.mj_model.light_diffuse[:] *= intensity_scale
            self.mj_model.light_specular[:] *= intensity_scale
            
            # 限制最大值为1.0
            self.mj_model.light_ambient[:] = np.clip(self.mj_model.light_ambient, 0, 1)
            self.mj_model.light_diffuse[:] = np.clip(self.mj_model.light_diffuse, 0, 1)
            self.mj_model.light_specular[:] = np.clip(self.mj_model.light_specular, 0, 1)
        
    def _randomize_table_height(self, table_config: Dict[str, Any]):
        """
        随机化桌面高度
        
        Args:
            table_config: 桌面高度随机化配置
        """
        table_name = table_config.get('table_name', 'table')

        if not hasattr(self, "table_pos0"):
            self.table_pos0 = self.mj_model.body(table_name).pos.copy()

        height_range = table_config.get('height_range', [0.0, 0.1])  # 默认0-10cm
        object_list = table_config.get('affected_objects', [])  # 受影响的物体列表
        
        print(f"🪑 桌面高度随机化: {table_name}")
        
        # 生成随机高度变化量
        if isinstance(height_range, list) and len(height_range) == 2:
            change_height = np.random.uniform(float(height_range[0]), float(height_range[1]))
        else:
            print(f"⚠️ 无效的高度范围配置: {height_range}, 使用默认0-10cm")
            change_height = np.random.uniform(0, 0.1)  # 默认0-10cm

        self.mj_model.body(table_name).pos[:] = self.table_pos0.copy()
        self.mj_model.body(table_name).pos[2] = self.table_pos0[2] - change_height

        for obj_name in object_list:
            try:
                self._object_pose(obj_name)[2] -= change_height
            except KeyError:
                print(f"⚠️ 未找到物体 '{obj_name}'，无法调整高度")
        
        print(f"   🎯 桌面高度随机化完成")
    
    def _randomize_textures(self, textures_config: Dict[str, Any]):
        """
        随机化材质纹理
        
        Args:
            textures_config: 材质随机化配置
        """
        print("🎨 开始材质随机化...")
        
        # 检查是否有纹理对象配置
        if 'objects' not in textures_config:
            print("⚠️ 材质配置中未找到objects字段")
            return
        
        objects_config = textures_config['objects']
        if not isinstance(objects_config, list):
            print("❌ textures.objects应该是一个列表")
            return
        
        # 遍历每个材质对象配置
        for obj_config in objects_config:
            self._randomize_single_texture(obj_config)
    
    def _randomize_single_texture(self, obj_config: Dict[str, Any]):
        """
        随机化单个材质对象
        
        Args:
            obj_config: 单个材质对象配置
        """
        texture_name = obj_config.get('name')
        if not texture_name:
            print("⚠️ 材质配置中缺少name字段")
            return
        
        try:
            self.mj_model.texture(texture_name)  # 确保纹理存在
        except KeyError:
            print(f"❌ 未找到纹理: {texture_name}")
            return
        
        mtl_type = obj_config.get('mtl_type', 'texture_1k')
        if mtl_type == "texture_1k":
            random_texture_data = get_random_texture()
        else:
            print(f"⚠️ 不支持的材质类型: {mtl_type}")
            return

        # 更新纹理数据
        self.mj_model.texture(texture_name).data = np.array(random_texture_data)

        if self.viewer is not None:
            self._update_texture_viewer(texture_name)
        else:
            print("⚠️ 未设置查看器，无法更新纹理显示")

        print(f"   ✅ 材质 {texture_name} 随机化成功")

    def _update_texture_viewer(self, texture_name: str):
        """
        更新查看器中的纹理显示
        
        Args:
            texture_name: 纹理名称
        """

        try:
            texture_id = self.mj_model.texture(texture_name).id  # 确保纹理存在
            if hasattr(self.viewer, 'update_texture'):
                self.viewer.update_texture(texture_id)
        except KeyError:
            print(f"❌ 未找到纹理: {texture_name}")
            return
