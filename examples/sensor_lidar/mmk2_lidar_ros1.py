import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.examples.ros1.mmk2_ros1_joy import MMK2ROS1JoyCtl

from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import create_lidar_single_line
from mujoco_lidar.examples.lidar_vis_ros1 import broadcast_tf, publish_scene, publish_point_cloud

if __name__ == "__main__":
    rospy.init_node('mmk2_lidar_node', anonymous=True)

    # 设置NumPy打印选项：精度为3位小数，禁用科学计数法，行宽为500字符
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    cfg = MMK2Cfg()
    cfg.render_set["width"] = 1280
    cfg.render_set["height"] = 720
    cfg.mjcf_file_path = "mjcf/mmk2_lidar.xml"
    cfg.use_gaussian_renderer = False

    # 初始化仿真环境
    exec_node = MMK2ROS1JoyCtl(cfg)
    exec_node.reset()

    # 创建TF广播者
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 创建激光雷达射线配置 - 参数10表示角度分辨率，2π表示完整360度扫描范围
    # 返回的rays_phi和rays_theta分别表示射线的俯仰角和方位角
    # 设置激光雷达数据发布频率为12Hz
    lidar_pub_rate = 12
    rays_theta, rays_phi = create_lidar_single_line(360, np.pi*2.)
    rospy.loginfo("rays_phi, rays_theta: {}, {}".format(rays_phi.shape, rays_theta.shape))

    # 创建MuJoCo激光雷达传感器对象，关联到当前渲染场景
    lidar_s2 = MjLidarWrapper(exec_node.mj_model, site_name="laser", backend="cpu")

    # Warm Start
    # 使用Taichi库进行光线投射计算，获取激光雷达点云数据
    lidar_s2.trace_rays(exec_node.mj_data, rays_theta, rays_phi)
    lidar_s2.get_hit_points()

    # 创建ROS发布者，用于将激光雷达数据发布为PointCloud2类型消息
    pub_lidar_s2 = rospy.Publisher('/mmk2/lidar_s2', PointCloud2, queue_size=1)

    def publish_scene_thread():
        # 创建场景可视化标记发布者
        rate = rospy.Rate(1)
        pub_scene = rospy.Publisher('/mujoco_scene', MarkerArray, queue_size=1)
        while exec_node.running and not rospy.is_shutdown():
            # 发布场景可视化标记
            publish_scene(pub_scene, exec_node.renderer.scene)
            rate.sleep()
    # 创建一个线程，用于发布场景可视化标记

    scene_pub_thread = threading.Thread(target=publish_scene_thread)
    scene_pub_thread.start()

    sim_step_cnt = 0
    lidar_pub_cnt = 0

    print("打开rviz并在其中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/mmk2/lidar_s2")
    print("3. 设置Fixed Frame为'world'")
    print("4. 添加MarkerArray显示，话题为/mujoco_scene")

    while exec_node.running and not rospy.is_shutdown():
        # 处理手柄操作输入
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)

        # 当累计的仿真时间（步数×时间步长×期望频率）超过已发布次数时，执行发布
        if sim_step_cnt * exec_node.delta_t * lidar_pub_rate > lidar_pub_cnt:
            lidar_pub_cnt += 1

            lidar_s2.trace_rays(exec_node.mj_data, rays_theta, rays_phi)
            points = lidar_s2.get_hit_points()
            publish_point_cloud(pub_lidar_s2, points, "laser")
           
            # 获取激光雷达位置和方向
            lidar_position = exec_node.mj_data.site("laser").xpos
            lidar_rotation_mat = exec_node.mj_data.site("laser").xmat.reshape(3, 3)
            lidar_orientation = Rotation.from_matrix(lidar_rotation_mat).as_quat()
            broadcast_tf(tf_broadcaster, "world", "laser", lidar_position, lidar_orientation)

        sim_step_cnt += 1

    scene_pub_thread.join()