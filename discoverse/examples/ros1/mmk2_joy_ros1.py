import os
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import json
import glfw
import xml.etree.ElementTree as ET

from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.airbot_play import AirbotPlayFIK
from discoverse.mmk2 import MMK2FIK
from discoverse.utils.joy_stick_ros1 import JoyTeleopRos1
from discoverse.utils import get_site_tmat, get_body_tmat

from discoverse import DISCOVERSE_ASSERT_DIR


def read_object_positions(xml_path):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 用于存储物体信息（name, pos）
    objects = []

    # 遍历所有的<worldbody>下的<body>标签
    for idx, body in enumerate(root.findall(".//worldbody/body")):
        name = body.get("name")  # 获取物品名称
        pos = body.get("pos")  # 获取位置属性
        if name and pos:
            # 打印物品名称及其位置，并存储物品信息
            objects.append((idx + 1, name, pos))  # 物品编号从1开始
            print(f"{idx + 1}. Object: {name}, Position: {pos}")

    return objects


class MMK2JOY(MMK2Base):
    arm_action_init_position = {
        "pick": {
            "l": np.array([0.254, 0.216, 1.069]),
            "r": np.array([0.254, -0.216, 1.069]),
        },
        "carry": {
            "l": np.array([0.254, 0.216, 1.069]),
            "r": np.array([0.254, -0.216, 1.069]),
        },
    }

    target_control = np.zeros(19)

    def __init__(self, config: MMK2Cfg):
        self.arm_action = config.init_key
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)

        self.lft_arm_target_pose = self.arm_action_init_position[self.arm_action][
            "l"
        ].copy()
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position[self.arm_action][
            "r"
        ].copy()
        self.rgt_end_euler = np.zeros(3)

        self.arm_fik = AirbotPlayFIK(
            urdf=os.path.join(
                DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"
            )
        )

        self.teleop = JoyTeleopRos1()

        self.objects = []  # 用于存储物体信息

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]
        self.lft_arm_target_pose = self.arm_action_init_position[self.arm_action][
            "l"
        ].copy()
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position[self.arm_action][
            "r"
        ].copy()
        self.rgt_end_euler = np.zeros(3)
        self.teleop.reset()

    def teleopProcess(self):
        linear_vel = 0.0
        angular_vel = 0.0
        if self.teleop.joy_cmd.buttons[4]:  # left arm
            tmp_lft_arm_target_pose = self.lft_arm_target_pose.copy()
            tmp_lft_arm_target_pose[0] += (
                self.teleop.joy_cmd.axes[7] * 0.1 / self.render_fps
            )
            tmp_lft_arm_target_pose[1] += (
                self.teleop.joy_cmd.axes[6] * 0.1 / self.render_fps
            )
            tmp_lft_arm_target_pose[2] += (
                self.teleop.joy_cmd.axes[1] * 0.1 / self.render_fps
            )

            delta_gripper = (
                (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5])
                * 1.0
                / self.render_fps
            )
            self.tctr_lft_gripper[0] += delta_gripper
            self.tctr_lft_gripper[0] = np.clip(self.tctr_lft_gripper[0], 0, 1)
            el = self.lft_end_euler.copy()
            el[0] += self.teleop.joy_cmd.axes[4] * 0.35 / self.render_fps
            el[1] += self.teleop.joy_cmd.axes[3] * 0.35 / self.render_fps
            el[2] += self.teleop.joy_cmd.axes[0] * 0.35 / self.render_fps
            try:
                self.tctr_left_arm[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(
                    tmp_lft_arm_target_pose,
                    self.arm_action,
                    "l",
                    self.tctr_slide[0],
                    self.tctr_left_arm,
                    Rotation.from_euler("zyx", el).as_matrix(),
                )
                self.lft_arm_target_pose[:] = tmp_lft_arm_target_pose
                self.lft_end_euler[:] = el
            except ValueError:
                print("Invalid left arm target position:", tmp_lft_arm_target_pose)

        if self.teleop.joy_cmd.buttons[5]:  # right arm
            tmp_rgt_arm_target_pose = self.rgt_arm_target_pose.copy()
            tmp_rgt_arm_target_pose[0] += (
                self.teleop.joy_cmd.axes[7] * 0.1 / self.render_fps
            )
            tmp_rgt_arm_target_pose[1] += (
                self.teleop.joy_cmd.axes[6] * 0.1 / self.render_fps
            )
            tmp_rgt_arm_target_pose[2] += (
                self.teleop.joy_cmd.axes[1] * 0.1 / self.render_fps
            )

            delta_gripper = (
                (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5])
                * 1.0
                / self.render_fps
            )
            self.tctr_rgt_gripper[0] += delta_gripper
            self.tctr_rgt_gripper[0] = np.clip(self.tctr_rgt_gripper[0], 0, 1)
            el = self.rgt_end_euler.copy()
            el[0] -= self.teleop.joy_cmd.axes[4] * 0.35 / self.render_fps
            el[1] += self.teleop.joy_cmd.axes[3] * 0.35 / self.render_fps
            el[2] -= self.teleop.joy_cmd.axes[0] * 0.35 / self.render_fps
            try:
                self.tctr_right_arm[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(
                    tmp_rgt_arm_target_pose,
                    self.arm_action,
                    "r",
                    self.tctr_slide[0],
                    self.tctr_right_arm,
                    Rotation.from_euler("zyx", el).as_matrix(),
                )
                self.rgt_arm_target_pose[:] = tmp_rgt_arm_target_pose
                self.rgt_end_euler[:] = el
            except ValueError:
                print("Invalid right arm target position:", tmp_rgt_arm_target_pose)

        if (not self.teleop.joy_cmd.buttons[4]) and (
            not self.teleop.joy_cmd.buttons[5]
        ):
            delta_height = (
                (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5])
                * 0.1
                / self.render_fps
            )
            if (
                self.tctr_slide[0] + delta_height
                < self.mj_model.joint("slide_joint").range[0]
            ):
                delta_height = (
                    self.mj_model.joint("slide_joint").range[0] - self.tctr_slide[0]
                )
            elif (
                self.tctr_slide[0] + delta_height
                > self.mj_model.joint("slide_joint").range[1]
            ):
                delta_height = (
                    self.mj_model.joint("slide_joint").range[1] - self.tctr_slide[0]
                )
            self.tctr_slide[0] += delta_height
            self.lft_arm_target_pose[2] -= delta_height
            self.rgt_arm_target_pose[2] -= delta_height

            self.tctr_head[0] += self.teleop.joy_cmd.axes[3] * 1.0 / self.render_fps
            self.tctr_head[1] -= self.teleop.joy_cmd.axes[4] * 1.0 / self.render_fps
            self.tctr_head[0] = np.clip(
                self.tctr_head[0],
                self.mj_model.joint("head_yaw_joint").range[0],
                self.mj_model.joint("head_yaw_joint").range[1],
            )
            self.tctr_head[1] = np.clip(
                self.tctr_head[1],
                self.mj_model.joint("head_pitch_joint").range[0],
                self.mj_model.joint("head_pitch_joint").range[1],
            )

            linear_vel = (
                1.0
                * self.teleop.joy_cmd.axes[1] ** 2
                * np.sign(self.teleop.joy_cmd.axes[1])
            )
            angular_vel = (
                2.0
                * self.teleop.joy_cmd.axes[0] ** 2
                * np.sign(self.teleop.joy_cmd.axes[0])
            )
        self.base_move(linear_vel, angular_vel)

    def base_move(self, linear_vel, angular_vel):
        self.tctr_base[0] = linear_vel
        self.tctr_base[1] = angular_vel

    def on_key(self, window, key, scancode, action, mods):
        super().on_key(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            if key == glfw.KEY_O:
                print(
                    "left_end position:",
                    get_site_tmat(self.mj_data, "lft_endpoint")[:3, 3],
                )
                print(
                    "right_end position:",
                    get_site_tmat(self.mj_data, "rgt_endpoint")[:3, 3],
                )
                self.save_state_to_json()

    def save_state_to_json(self):
        """保存机器人的当前状态到 JSON 文件"""
        self.objects = read_object_positions(self.mjcf_file)

        # 提示用户选择物体编号
        if self.objects:
            print("\n请选择一个物体的编号:")
            try:
                selected_idx = int(input(f"请输入编号（1 到 {len(self.objects)}）：")) - 1
            except ValueError:
                print("无效编号，保存失败。")
                return
            if 0 <= selected_idx < len(self.objects):
                selected_object = self.objects[selected_idx]
                selected_name = selected_object[1]
                selected_pos = get_body_tmat(self.mj_data, selected_name)[
                    :3, 3
                ]  # 物体的位置信息
                print(f"选择了物体: {selected_name}, 位置: {selected_pos}")
            else:
                print("无效编号，保存失败。")
                return
        else:
            print("未找到任何物体。")
            return

        select_arm = input("请选择要移动的机械臂（a/l/r）：")
        if not select_arm in {"a", "l", "r"}:
            print(("无效输入，保存失败。"))
            return

        try:
            delay_time_s = float(input("请输入延迟时间（秒）："))
        except ValueError:
            delay_time_s = 0.0

        # 获取机器人的当前状态，并调整为相对于物体的坐标
        state_data = {
            "object_name": selected_name,  # 使用选择的物体名称
            "left_arm": {
                "position_object_local": [
                    round(coord - selected_pos[idx], 3)
                    for idx, coord in enumerate(
                        get_site_tmat(self.mj_data, "lft_endpoint")[:3, 3]
                    )
                ],
                "rotation_robot_local": [round(val, 3) for val in self.lft_end_euler],
                "gripper": round(self.tctr_lft_gripper[0], 3),
                "movement": "stop" if select_arm == "r" else "move",
            },
            "right_arm": {
                "position_object_local": [
                    round(coord - selected_pos[idx], 3)
                    for idx, coord in enumerate(
                        get_site_tmat(self.mj_data, "rgt_endpoint")[:3, 3]
                    )
                ],
                "rotation_robot_local": [round(val, 3) for val in self.rgt_end_euler],
                "gripper": round(self.tctr_rgt_gripper[0], 3),
                "movement": "stop" if select_arm == "l" else "move",
            },
            "slide": [round(self.tctr_slide[0], 3)],
            "head": [round(self.tctr_head[0], 3), round(self.tctr_head[1], 3)],
            "delay_s": delay_time_s,
        }

        # 定义 JSON 文件保存路径
        json_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"scheme_{os.path.basename(self.mjcf_file).split('.')[0]}.json",
        )

        # 如果文件已存在，则加载已有数据并追加新状态，否则创建一个新文件
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        # 添加当前状态到列表中
        data.append(state_data)

        # 将更新后的数据写入 JSON 文件
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

        print("当前状态已保存到 JSON 文件。")

    def printMessage(self):
        super().printMessage()

        print("    lta local = {}".format(self.lft_arm_target_pose))
        print("    rta local = {}".format(self.rgt_arm_target_pose))
        print("       euler  = {}".format(self.lft_end_euler))
        print("       euler  = {}".format(self.rgt_end_euler))


if __name__ == "__main__":
    rospy.init_node("mmk2_mujoco_node", anonymous=True)

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()

    cfg.init_key = "pick"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = None
    cfg.obs_depth_cam_id = None

    cfg.render_set = {"fps": 30, "width": 1920, "height": 1080}
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/plate_coffeecup.xml"

    exec_node = MMK2JOY(cfg)
    exec_node.reset()

    while exec_node.running:
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)
