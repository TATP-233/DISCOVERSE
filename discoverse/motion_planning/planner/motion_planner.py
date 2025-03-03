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

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import mujoco

from planner.mujoco_ompl_plugin import MJOMPL, MJOMPLRobot, CollisionDetector

DEFAULT_PLANNING_TIME = 10
DEFAULT_INTERPOLATE_NUM = 100


class AirbotPlayCollisionDetector:
    def __init__(self, mj_model) -> None:
        self.mj_model = mj_model
        self.mj_data_col = mujoco.MjData(self.mj_model)

    def check_collision(self, qpos):
        return False

    def update_collision_model(self, obj_name):
        pass


AirbotPlayMotionPlannerDefaultConfig = {
    "edge_resolution": 0.005,
    "allowed_planning_time": 1.0,
    "interpolate_num": 10,
    "planner": "RRT",
    "planner_params": {},
}


class AirbotPlayMotionPlanner:
    def __init__(self, sim_node, cfg=AirbotPlayMotionPlannerDefaultConfig):

        self.sim_node = sim_node
        self.task_instances = sim_node.task_instances
        self.cfg = cfg

        # Construct OMPL planner
        flexible_jqpos_idx = [0, 1, 2, 3, 4, 5]
        flexible_jctrl_idx = [0, 1, 2, 3, 4, 5]
        mj_ompl_robot = MJOMPLRobot(sim_node, flexible_jqpos_idx, flexible_jctrl_idx)
        self.cls_det = AirbotPlayCollisionDetector(sim_node.mj_model)
        self.mj_ompl = MJOMPL(mj_ompl_robot, self.cls_det, self.cfg["edge_resolution"])
        # setup planner
        self.mj_ompl.set_planner(self.cfg["planner"])

    def update_collision_detector(self):
        robot_state = self.sim_node.getRobotState()
        if robot_state[1] != None:
            self.cls_det.update_collision_model(robot_state[1])

    def get_motion_plan(self, start_pos, target_pose):
        solu_list = []
        solved, solu_list = self.mj_ompl.plan_start_goal(
            start_pos,
            target_pose,
            self.cfg["allowed_planning_time"],
            self.cfg["interpolate_num"],
        )

        if solved:
            return solu_list
        else:
            print("No feasible motion plan!")
            return []
