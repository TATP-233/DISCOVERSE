import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray

from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.examples.ros2.mmk2_ros2_joy import MMK2ROS2JoyCtl

from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import create_lidar_single_line
from mujoco_lidar.mj_lidar_utils import create_marker_from_geom

def publish_scene(publisher, mj_scene, frame_id, stamp):
    """将MuJoCo场景发布为ROS可视化标记数组"""
    marker_array = MarkerArray()

    # 记录当前使用的标记ID
    current_id = 0

    # 创建每个几何体的标记
    for i in range(mj_scene.ngeom):
        geom = mj_scene.geoms[i]
        # 创建标记并返回一个标记列表
        markers = create_marker_from_geom(geom, current_id, frame_id)

        # 添加所有返回的标记到标记数组
        for marker in markers:
            # 在ROS2中，需要设置stamp为ROS2的时间类型
            marker.header.stamp = stamp
            marker_array.markers.append(marker)
            current_id += 1

    # 发布标记数组
    publisher.publish(marker_array)


def publish_point_cloud(publisher, points, frame_id, stamp):
    """将点云数据发布为ROS PointCloud2消息"""

    # 定义点云字段
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]

    # 添加强度值
    if len(points.shape) == 2:
        # 如果是(N, 3)形状，转换为(3, N)以便处理
        points_transposed = points.T if points.shape[1] == 3 else points

        if points_transposed.shape[0] == 3:
            # 添加强度通道
            points_with_intensity = np.vstack([
                points_transposed, 
                np.ones(points_transposed.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points_transposed
    else:
        # 如果点云已经是(3, N)形状
        if points.shape[0] == 3:
            points_with_intensity = np.vstack([
                points, 
                np.ones(points.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points

    # 创建ROS2 PointCloud2消息
    pc_msg = PointCloud2()
    pc_msg.header.frame_id = frame_id
    pc_msg.header.stamp = stamp
    pc_msg.fields = fields
    pc_msg.is_bigendian = False
    pc_msg.point_step = 16  # 4 个 float32 (x,y,z,intensity)
    pc_msg.row_step = pc_msg.point_step * points_with_intensity.shape[1]
    pc_msg.height = 1
    pc_msg.width = points_with_intensity.shape[1]
    pc_msg.is_dense = True

    # 转置回(N, 4)格式并转换为字节数组
    pc_msg.data = np.transpose(points_with_intensity).astype(np.float32).tobytes()

    publisher.publish(pc_msg)


def broadcast_tf(broadcaster, parent_frame, child_frame, translation, rotation, stamp):
    """广播TF变换"""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])

    t.transform.rotation.x = float(rotation[0])
    t.transform.rotation.y = float(rotation[1])
    t.transform.rotation.z = float(rotation[2])
    t.transform.rotation.w = float(rotation[3])

    broadcaster.sendTransform(t)

if __name__ == "__main__":
    rclpy.init()

    # 设置NumPy打印选项：精度为3位小数，禁用科学计数法，行宽为500字符
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    cfg = MMK2Cfg()
    cfg.render_set["width"] = 1280
    cfg.render_set["height"] = 720
    cfg.init_key = "pick"
    cfg.mjcf_file_path = "mjcf/mmk2_lidar.xml"
    cfg.use_gaussian_renderer = False

    # 初始化仿真环境
    exec_node = MMK2ROS2JoyCtl(cfg)
    exec_node.reset()

    # 创建TF广播者
    tf_broadcaster = TransformBroadcaster(exec_node)

    # 创建激光雷达射线配置 - 参数360表示水平分辨率，2π表示完整360度扫描范围
    # 返回的rays_phi和rays_theta分别表示射线的俯仰角和方位角
    # 设置激光雷达数据发布频率为12Hz
    lidar_pub_rate = 12
    rays_theta, rays_phi = create_lidar_single_line(360, np.pi*2.)
    exec_node.get_logger().info("rays_phi, rays_theta: {}, {}".format(rays_phi.shape, rays_theta.shape))

    # 创建MuJoCo激光雷达传感器对象，关联到当前渲染场景
    # enable_profiling=False表示不启用性能分析，verbose=False表示不输出详细日志
    lidar_s2 = MjLidarWrapper(exec_node.mj_model, site_name="laser", backend="cpu", args={'bodyexclude': exec_node.mj_model.body("agv_link").id})
    # Warm Start
    # 使用Taichi库进行光线投射计算，获取激光雷达点云数据
    lidar_s2.trace_rays(exec_node.mj_data, rays_theta, rays_phi)
    lidar_s2.get_hit_points()
    
    # 创建ROS发布者，用于将激光雷达数据发布为PointCloud2类型消息
    pub_lidar_s2 = exec_node.create_publisher(PointCloud2, '/mmk2/lidar_s2', 1)

    def publish_scene_thread():
        # 创建ROS发布者，用于将MuJoCo场景发布为MarkerArray类型消息
        rate = exec_node.create_rate(1)
        pub_scene = exec_node.create_publisher(MarkerArray, '/mujoco_scene', 1)
        while exec_node.running and rclpy.ok():
            stamp = exec_node.get_clock().now().to_msg()
            publish_scene(pub_scene, exec_node.renderer.scene, "world", stamp)
            rate.sleep()

    # 创建一个线程，用于发布场景可视化标记
    scene_pub_thread = threading.Thread(target=publish_scene_thread)
    scene_pub_thread.start()

    sim_step_cnt = 0
    lidar_pub_cnt = 0

    print("打开rviz2并在其中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/mmk2/lidar_s2")
    print("3. 设置Fixed Frame为'world'")
    print("4. 添加MarkerArray显示，话题为/mujoco_scene")

    while exec_node.running and rclpy.ok():
        # 处理ROS消息
        rclpy.spin_once(exec_node, timeout_sec=0)
        
        # 处理手柄操作输入
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)

        # 当累计的仿真时间（步数×时间步长×期望频率）超过已发布次数时，执行发布
        if sim_step_cnt * exec_node.delta_t * lidar_pub_rate > lidar_pub_cnt:
            lidar_pub_cnt += 1

            lidar_s2.trace_rays(exec_node.mj_data, rays_theta, rays_phi)
            points = lidar_s2.get_hit_points()

            stamp = exec_node.get_clock().now().to_msg()
            publish_point_cloud(pub_lidar_s2, points, "laser", stamp)

            # 获取激光雷达位置和方向
            lidar_position = exec_node.mj_data.site("laser").xpos
            lidar_rotation_mat = exec_node.mj_data.site("laser").xmat.reshape(3, 3)
            lidar_orientation = Rotation.from_matrix(lidar_rotation_mat).as_quat()
            broadcast_tf(tf_broadcaster, "world", "laser", lidar_position, lidar_orientation, stamp)

        sim_step_cnt += 1

    # 清理资源
    exec_node.destroy_node()
    rclpy.shutdown()
    scene_pub_thread.join()
