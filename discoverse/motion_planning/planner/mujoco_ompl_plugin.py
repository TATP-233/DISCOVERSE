from os.path import abspath, dirname, join
import os
import sys

base_dir = os.getcwd()
ompl_dir = join(base_dir, "../")
print(ompl_dir)
sys.path.insert(0, join(ompl_dir, "ompl/py-bindings"))
sys.path.insert(1, base_dir)
print(sys.path)
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import json
import yaml
import time
from itertools import product
from simple_pid import PID
import math
import copy
import numpy as np
import mediapy

import mujoco

DEFAULT_PLANNING_TIME = 10
INTERPOLATE_NUM = 100


class MJOMPLRobot:
    """
    Input:
    """

    def __init__(self, mj_node, flexible_jqpos_idx=[], flexible_jctrl_idx=[]) -> None:

        # Public attributes
        self.mj_node = mj_node
        self.pos_dim = self.mj_node.njq
        self.flexible_jqpos_idx = flexible_jqpos_idx
        self.flexible_jctrl_idx = flexible_jctrl_idx
        self.flexible_jctrl_num = len(flexible_jctrl_idx)

        self.joint_bounds = []
        self.joint_bounds = self.get_joint_bounds()
        # print("Current ctrl state of robot", self.get_cur_ctrl_state())
        # print("Current qpos of robot", self.get_cur_qpos())
        # print("Joint bounds: {}".format(self.joint_bounds))

    def get_ompl_state_space(self):
        return ob.RealVectorStateSpace(self.flexible_jctrl_num)

    def get_ompl_space_bounds(self):
        bounds = ob.RealVectorBounds(self.flexible_jctrl_num)
        for i, bound in enumerate(self.joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        return bounds

    def get_joint_bounds(self):
        """
        Get joint bounds.
        By default, read from pybullet
        """
        for joint_id in self.flexible_jctrl_idx:
            low = self.mj_node.mj_model.actuator_ctrlrange[joint_id][0]  # low bounds
            high = self.mj_node.mj_model.actuator_ctrlrange[joint_id][1]  # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
            else:
                print("invalid joint range, index = ", joint_id)
        return self.joint_bounds

    def get_cur_ctrl_state(self):
        return self.mj_node.mj_data.ctrl[: self.mj_node.njctrl].copy()

    def get_cur_qpos(self):
        return self.mj_node.mj_data.qpos[: self.mj_node.njq].copy()

    def reset(self):
        """
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        """
        self.mj_node.resetState()

    def set_robot_qpos(self, qpos):
        self.mj_node.mj_data.qpos = qpos

    def execute(self, path, vedio_name="mmk2_dual_arm_plan"):
        """
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
        """
        path = np.array(path)
        # 重置环境
        obs = self.mj_node.reset()
        action = self.mj_node.init_ctrl.copy()
        obsss = []
        steps = path.shape[0]
        for i in range(steps):
            action[self.flexible_jctrl_idx] = path[i]
            obs, _, _, _, _ = self.mj_node.step(action)
            obsss.append(obs["img"][-1].copy())
            img_show = cv2.cvtColor(obs["img"][-1], cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img_show)
            key = cv2.waitKey(1000 // 30)  # 30fps
            if key == 27:
                break

        cv2.destroyAllWindows()

        mediapy.write_video(vedio_name + ".mp4", obsss, fps=30)


class CollisionDetector:
    # def __init__(self, mjcf_file) -> None:
    # self.mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
    def __init__(self, mj_model) -> None:
        self.mj_model = mj_model
        self.mj_data_col = mujoco.MjData(self.mj_model)

    def check_collision(self, qpos):
        """
        Check if the robot is in collision with the environment.
        TODO ： self collision check
        Args:
            qpos (np.ndarray): The joint positions of the robot.
        Returns:
            bool: True if the robot is in collision with the environment.
        """
        self.mj_data_col.qpos[:] = qpos[:]
        mujoco.mj_forward(self.mj_model, self.mj_data_col)
        # 0是地面id 计算除了地面以外的碰撞
        return bool(np.where(self.mj_data_col.contact.geom[:, 0] != 0)[0].shape[0])


class MJOMPL:
    def __init__(
        self,
        mj_ompl_robot,
        collision_detector,
        edge_resolution=0.005,
        planner_param_file=None,
    ) -> None:
        """
        Args
            mj_ompl_robot: A MJOMPLRobot instance.
            collision_detector: A CollisionDetector instance
        """
        self.mj_robot = mj_ompl_robot
        self.collision_detector = collision_detector

        # Initialize OMPL state space
        self.space = (
            self.mj_robot.get_ompl_state_space()
        )  # dim = num of joints to control
        self.space.setBounds(self.mj_robot.get_ompl_space_bounds())

        # OMPL simple setup
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        self.si.setStateValidityCheckingResolution(edge_resolution)
        self.set_planner("RRTstar")  # RRT by default

    def is_state_valid(self, ompl_state):
        """
        Given an OMPL state, check if the robot is in collision with the environment.
        Args:
            state (np.ndarray): The OMPL state with dimensions equal to self.mj_robot.num_joint_to_ctrl
        Returns:
            bool:
        """
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set
        state = self.state_to_list(ompl_state)
        qpos = self.mj_robot.get_cur_qpos()
        for i, s in enumerate(state):
            qpos[self.mj_robot.flexible_jqpos_idx[i]] = s
        return not self.collision_detector.check_collision(qpos)

    def set_planner(self, planner_name, param_file_path=None):
        """
        Note: Add your planner here!!
        """
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "RRTXstatic":
            self.planner = og.RRTXstatic(self.ss.getSpaceInformation())
        elif planner_name == "LLPTstatic":
            self.planner = og.LLPTstatic(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        if param_file_path != None:
            with open(param_file_path) as stream:
                try:
                    plannerParams = yaml.safe_load(stream)
                    params = self.planner.params()
                    for param_name, param_value in plannerParams[planner_name].items():
                        params[param_name].setValue(str(param_value))
                except yaml.YAMLError as exc:
                    print(exc)

        self.ss.setPlanner(self.planner)
        return self.planner

    def set_start_goal(self, start_qpos, goal_qpos):
        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i, joint_idx in enumerate(self.mj_robot.flexible_jqpos_idx):
            s[i] = start_qpos[joint_idx]
            g[i] = goal_qpos[joint_idx]

        print("Is start state valid ", self.is_state_valid(s))
        print("Is goal state valid ", self.is_state_valid(g))

        self.ss.setStartAndGoalStates(s, g)

    def plan_start_goal(
        self,
        start_qpos,
        goal_qpos,
        allowed_time=DEFAULT_PLANNING_TIME,
        interpolation_num=INTERPOLATE_NUM,
    ):
        """
        plan a path to goal from the given robot start state
        """
        # print("start_planning")
        # print(self.planner.params())

        # orig_robot_qpos= self.mj_robot.get_cur_qpos()

        # set the start and goal states;
        self.set_start_goal(start_qpos, goal_qpos)

        # attempt to solve the problem within allowed planning time
        self.ss.solve(allowed_time)
        solved = self.ss.haveExactSolutionPath()
        res = False
        sol_path_list = []
        if solved:
            # print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            print(sol_path_geometric)
            # sol_path_geometric.interpolate(INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            res = True
            for sol_path in sol_path_list:
                if not self.is_state_valid(sol_path):
                    res = False
                    sol_path_list = []
                    break
        else:
            print("No solution found")

        # reset robot state
        # self.mj_robot.reset()
        self.ss.clear()
        return res, sol_path_list

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME):
        """
        plan a path to gaol from current robot state
        """
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time)

    def execute_step(self, goal_pos):
        """
        Execute a ctrl_pos
        Args:
            goal_pos: x and y of goal state
        """
        self.mj_robot.mj_node.mj_data.qpos[self.mj_robot.flexible_jqpos_idx] = goal_pos
        mujoco.mj_forward(self.mj_robot.mj_node.mj_model, self.mj_robot.mj_node.mj_data)
        return

    def state_to_list(self, state):
        return [state[i] for i in range(self.mj_robot.flexible_jctrl_num)]


if __name__ == "__main__":
    import cv2  # conda install -c conda-forge opencv
    from discoverse.envs.mmk2_base import MMK2Cfg, MMK2Base

    TWO_ARM_PLANNING = True
    LEFT_ARM_PLANNING = True
    planner_param_file = (
        base_dir + "/discoverse/motion_planning/params/mmk2_planning_param.yaml"
    )
    EDGE_RESOLUTION = 0.005
    ALLOWED_PLANNING_TIME = 30
    INTERPOLATE_NUM = 50

    # 配置文件
    cfg = MMK2Cfg()
    cfg.render_set["height"] = 480  # 渲染窗口高度
    cfg.render_set["width"] = 640  # 渲染窗口宽度
    cfg.mjcf_file_path = "mjcf/exhibition.xml"  # mjcf模型文件路径 models路径下
    cfg.use_gaussian_renderer = False  # 不使用高斯渲染器
    cfg.headless = True  # 不显示窗口
    cfg.obs_camera_id = [-1]  # 相机id
    cfg.init_key = "front_table"  # 初始化位姿

    # 创建环境
    env = MMK2Base(cfg)
    env.reset()
    # 设置渲染标志 mjVIS_CONTACTFORCE 显示接触力
    env.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # 设置相机视角
    env.free_camera.lookat[:] = env.getObjPose("head_cam")[0]
    env.free_camera.lookat[2] -= 0.5
    env.free_camera.distance = 2
    """
    njqpos=28
    [0:7]-base; 7-lft_wheel; 8-rgt_wheel; 9-slide; 10-head_yaw"; 11-head_pitch; [12:20]-lft_arm ; [20:28]-rgt_arm

    njctrl=19
    0-forward; 1-turn; 2-lift; 3-yaw; 4-pitch; [5:12]-lft_arm; [12:19]-rgt_arm
    """
    left_arm_jqpos_idx = [12, 13, 14, 15, 16, 17, 18]
    left_arm_jctrl_idx = [5, 6, 7, 8, 9, 10, 11]
    right_arm_jqpos_idx = [20, 21, 22, 23, 24, 26, 26]
    right_arm_jctrl_idx = [12, 13, 14, 15, 16, 17, 18]

    if TWO_ARM_PLANNING:
        flexible_jqpos_idx = left_arm_jqpos_idx + right_arm_jqpos_idx
        flexible_jctrl_idx = left_arm_jctrl_idx + right_arm_jctrl_idx
    elif LEFT_ARM_PLANNING:
        flexible_jqpos_idx = left_arm_jqpos_idx
        flexible_jctrl_idx = left_arm_jctrl_idx
    else:
        flexible_jqpos_idx = right_arm_jqpos_idx
        flexible_jctrl_idx = right_arm_jctrl_idx

    mj_ompl_robot = MJOMPLRobot(env, flexible_jqpos_idx, flexible_jctrl_idx)
    cls_det = CollisionDetector(env.mj_model)
    mj_ompl = MJOMPL(mj_ompl_robot, cls_det, EDGE_RESOLUTION, planner_param_file)

    start_qpos = mj_ompl_robot.get_cur_qpos()
    start_ctrl_state = mj_ompl_robot.get_cur_ctrl_state()
    print("Start qpos is", start_qpos[flexible_jqpos_idx])
    print("Start ctrl_state is", start_ctrl_state[flexible_jctrl_idx])

    # set goal position
    goal_qpos = start_qpos.copy()
    if TWO_ARM_PLANNING:
        left_arm_goal_qpos = [-2, -1.4, 2.7, 0.5, 0.5, 0, 0]
        right_arm_goal_qpos = [2, -1.4, 2.7, 0.5, 0.5, 0, 0]
        goal_qpos[flexible_jqpos_idx] = left_arm_goal_qpos + right_arm_goal_qpos
        print("Goal qpos is", goal_qpos[flexible_jqpos_idx])
    elif LEFT_ARM_PLANNING:
        left_arm_goal_qpos = [-2, -1.4, 2.7, 0.5, 0.5, 0, 0]
        goal_qpos[flexible_jqpos_idx] = left_arm_goal_qpos
        print("Goal qpos is", goal_qpos[flexible_jqpos_idx])
    else:
        right_arm_goal_qpos = [2, 0.1, 1, 0, 1, 0, 0]
        goal_qpos[flexible_jqpos_idx] = right_arm_goal_qpos
        print("Goal qpos is", goal_qpos[flexible_jqpos_idx])

    # initialize environment
    env.mj_model.body("red_box").pos[0] = -1.2
    env.mj_model.body("red_box").pos[1] = -1.8

    # setup planner
    mj_ompl.set_planner("LLPTstatic", planner_param_file)

    # start planning
    solved, solu_list = mj_ompl.plan_start_goal(
        start_qpos, goal_qpos, ALLOWED_PLANNING_TIME, INTERPOLATE_NUM
    )
    # print(solu_list)

    if solved:
        mj_ompl_robot.execute(solu_list)

    # TODO: test in more scenarios
