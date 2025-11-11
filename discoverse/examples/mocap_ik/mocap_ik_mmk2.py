import os
import sys
import time
import platform
import argparse
import mujoco
import mujoco.viewer
import numpy as np

import discoverse
from discoverse.examples.mocap_ik.mocap_ik_utils import \
    mj_quat2mat, \
    add_mocup_body_to_mjcf, \
    move_mocap_to_frame, \
    generate_mocap_xml, \
    generate_mocap_sensor_xml

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.robots import MMK2IK

if __name__ == "__main__":  
    """
    MMK2机器人的仿真主程序
    
    该程序创建一个MMK2机器人模型的MuJoCo仿真环境，添加运动捕捉(mocap)目标，
    并使用逆运动学(IK)控制机器人的双臂跟踪目标位置和姿态。
    """
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)

    parser = argparse.ArgumentParser(
        description="Airbot Play 机器人MuJoCo仿真主程序\n"
                    "用法示例：\n"
                    "  python mocap_ik_airbot_play.py [--mjcf]\n"
                    "参数说明：\n"
                    "  mjcf  (可选) 指定MJCF模型文件路径，若不指定则使用默认模型。"
    )
    parser.add_argument(
        "-m",
        "--mjcf",
        type=str,
        default=None,
        help="输入MJCF文件的路径（可选）。如未指定，则使用默认的airbot_play_floor.xml。"
    )
    args = parser.parse_args()
    mjcf_path = args.mjcf

    # 检查是否在macOS上运行并给出适当的提示
    if platform.system() == "Darwin":
        print("\n===================================================")
        print("注意: 在macOS上运行MuJoCo查看器需要使用mjpython")
        print("请使用以下命令运行此脚本:")
        print(f"mjpython {' '.join(sys.argv)}")
        print("===================================================\n")
        
        user_input = input("是否继续尝试启动查看器? (y/n): ")
        if user_input.lower() != 'y':
            print("退出程序。")
            sys.exit(0)
    # 打印MuJoCo查看器的使用提示信息
    print("\n===================== MuJoCo 查看器使用说明 =====================")
    print("1. 双击选中机械臂末端的绿色方块，按下ctrl和鼠标左键，拖动鼠标可以旋转机械臂")
    print("   按下ctrl和鼠标右键，拖动鼠标可以平移机械臂末端")
    print("2. 按 Tab 键切换左侧 UI 的可视化界面。")
    print("   按 Shift+Tab 键切换右侧 UI 的可视化界面。")
    print("3. 在左侧 UI 中点击 'Copy State' 可以当前的机器人状态复制到剪贴板。")
    print("4. 在右侧 UI 的 control 面板中可以调节 gripper 滑块控制夹爪的开合。")
    print("================================================================\n")
    # 设置numpy输出格式
    np.set_printoptions(precision=5, suppress=True, linewidth=500)

    # 加载MMK2机器人模型的MJCF文件
    # 加载Airbot Play机器人模型的MJCF文件
    if mjcf_path is None:
        mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "mmk2_floor.xml")
    else:
        if not os.path.exists(mjcf_path):
            paths = [
                os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", mjcf_path),
                os.path.join(DISCOVERSE_ASSETS_DIR, mjcf_path),
                os.path.join(os.getcwd(), mjcf_path),
            ]
            for path in paths:
                if os.path.exists(path) and os.path.isfile(path) and (path.endswith(".xml") or path.endswith(".mjb")):
                    mjcf_path = path
                    break
        if mjcf_path is None:
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    print("mjcf_path : " , mjcf_path)

    # 设置渲染帧率
    render_fps = 50

    # 左臂目标（mocap）名称
    lft_mocap_name = "lft_target"
    lft_mocap_box_name = lft_mocap_name + "_box"

    # 右臂目标（mocap）名称
    rgt_mocap_name = "rgt_target"
    rgt_mocap_box_name = rgt_mocap_name + "_box"

    # 生成左右臂的mocap刚体XML元素
    mocap_body_elements = []
    mocap_body_elements.append(generate_mocap_xml(lft_mocap_name))
    mocap_body_elements.append(generate_mocap_xml(rgt_mocap_name))

    # 生成左右臂的mocap传感器XML元素
    sensor_elements = []
    sensor_elements.extend(generate_mocap_sensor_xml(lft_mocap_name, ref_name="base_link", ref_type="site"))
    sensor_elements.extend(generate_mocap_sensor_xml(rgt_mocap_name, ref_name="base_link", ref_type="site"))

    # 将mocap刚体和传感器添加到模型中
    mj_model = add_mocup_body_to_mjcf(mjcf_path, mocap_body_elements, sensor_elements, keep_tmp_xml=True)
    # 计算渲染间隔，确保按照指定帧率渲染
    render_gap = int(1.0 / render_fps / mj_model.opt.timestep)

    # 创建MuJoCo数据实例
    mj_data = mujoco.MjData(mj_model)

    # 获取提升关节位置传感器ID
    slide_sensor_id = mj_model.sensor("lift_pos").adr[0]

    # 获取左臂mocap位置和方向传感器ID
    lft_mocap_posi_sensor_id = mj_model.sensor(f"{lft_mocap_name}_pos").adr[0]
    lft_mocap_quat_sensor_id = mj_model.sensor(f"{lft_mocap_name}_quat").adr[0]

    # 获取右臂mocap位置和方向传感器ID
    rgt_posi_sensor_id = mj_model.sensor(f"{rgt_mocap_name}_pos").adr[0]
    rgt_quat_sensor_id = mj_model.sensor(f"{rgt_mocap_name}_quat").adr[0]

    # 获取左臂关节位置传感器ID
    lft_armjoint_sensor_ids = [mj_model.sensor(f"lft_joint{i+1}_pos").adr[0] for i in range(6)]
    # 获取右臂关节位置传感器ID
    rgt_armjoint_sensor_ids = [mj_model.sensor(f"lft_joint{i+1}_pos").adr[0] for i in range(6)]

    # 获取左臂关节控制器ID
    lft_armjoint_control_ids = [mj_model.actuator(f"lft_joint{i+1}").id for i in range(6)]
    # 获取右臂关节控制器ID
    rgt_armjoint_control_ids = [mj_model.actuator(f"rgt_joint{i+1}").id for i in range(6)]

    # 创建MMK2逆运动学求解器
    mmk2ik = MMK2IK()

    try:
        # 启动MuJoCo查看器
        with mujoco.viewer.launch_passive(
            mj_model, mj_data, 
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running():
                # 记录步骤开始时间
                step_start = time.time()

                # 执行渲染间隔次数的物理仿真步骤
                for _ in range(render_gap):
                    mujoco.mj_step(mj_model, mj_data)

                # 获取提升关节位置
                slide_joint_pos = mj_data.sensordata[slide_sensor_id]

                # 获取左臂目标位置和旋转矩阵
                t_lft_posi = mj_data.sensordata[lft_mocap_posi_sensor_id:lft_mocap_posi_sensor_id+3]
                t_lft_rmat = mj_quat2mat(mj_data.sensordata[lft_mocap_quat_sensor_id:lft_mocap_quat_sensor_id+4])

                # 获取右臂目标位置和旋转矩阵
                t_rgt_posi = mj_data.sensordata[rgt_posi_sensor_id:rgt_posi_sensor_id+3]
                t_rgt_rmat = mj_quat2mat(mj_data.sensordata[rgt_quat_sensor_id:rgt_quat_sensor_id+4])

                # 获取当前左臂关节位置作为参考值
                lft_q_ref = mj_data.sensordata[lft_armjoint_sensor_ids].copy()
                # 获取当前右臂关节位置作为参考值
                rgt_q_ref = mj_data.sensordata[rgt_armjoint_sensor_ids].copy()
                
                try:
                    # 计算左臂逆运动学解，获取关节角度
                    jq = mmk2ik.armIK_wrt_footprint(t_lft_posi, t_lft_rmat, "l", slide_joint_pos, lft_q_ref)
                    # 控制左臂关节执行计算出的角度
                    mj_data.ctrl[lft_armjoint_control_ids] = jq
                    # 设置左臂目标框为绿色（表示IK计算成功）
                    mj_model.geom(lft_mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
                except ValueError as e:
                    # 捕获逆运动学计算异常
                    # print(e)
                    # 将左臂目标移动到左臂末端执行器位置
                    move_mocap_to_frame(mj_model, mj_data, lft_mocap_name, "lft_endpoint", "site")
                    # 设置左臂目标框为红色（表示IK计算失败）
                    mj_model.geom(lft_mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)

                try:
                    # 计算右臂逆运动学解，获取关节角度
                    jq = mmk2ik.armIK_wrt_footprint(t_rgt_posi, t_rgt_rmat, "r", slide_joint_pos, rgt_q_ref)
                    # 控制右臂关节执行计算出的角度
                    mj_data.ctrl[rgt_armjoint_control_ids] = jq
                    # 设置右臂目标框为绿色（表示IK计算成功）
                    mj_model.geom(rgt_mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
                except ValueError as e:
                    # 捕获逆运动学计算异常
                    # print(e)
                    # 将右臂目标移动到右臂末端执行器位置
                    move_mocap_to_frame(mj_model, mj_data, rgt_mocap_name, "rgt_endpoint", "site")
                    # 设置右臂目标框为红色（表示IK计算失败）
                    mj_model.geom(rgt_mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)

                # 同步查看器状态
                viewer.sync()
                
                # 计算下一步开始前需要等待的时间，保证帧率稳定
                time_until_next_step = (1. / render_fps) - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except RuntimeError as e:
        if "mjpython" in str(e) and platform.system() == "Darwin":
            print("\n错误: 在macOS上必须使用mjpython运行此脚本")
            print("请使用以下命令:")
            print(f"mjpython {' '.join(sys.argv)}")
        else:
            print(f"运行时错误: {e}")