from os.path import abspath, dirname, join
import os
import sys

base_dir = os.getcwd()
motion_planning_dir = join(base_dir, "discoverse/motion_planning")
sys.path.insert(0, motion_planning_dir)

import datetime
import numpy as np
import argparse, shutil, logging

from task_instances.create_sim_node import AirbotPlaySimNode
from planner.llm_task_planner import LLMTaskPlanner
from planner.motion_planner import AirbotPlayMotionPlanner
from utils import setup_global_logger

CONTROL_PRECISION = 1e-2


def executeTaskPlan(sim_node, task_plan, motion_planner=None):
    if motion_planner == None:
        print(
            "No motion planner specified. Using geometric interpolation to generate motions."
        )

    success = True
    obs = sim_node.reset()
    data_idx = 0
    obs_lst, act_lst = [], []
    process_list = []

    action = sim_node.init_joint_pose[: sim_node.nj].copy()
    action[6] *= 25.0
    init_control = action.copy()
    tarjq = init_control.copy()

    for task_command_i, task_command in enumerate(task_plan):
        print(sim_node.robot_state)
        print(task_command)
        if task_command[0] == "close_gripper" or task_command[0] == "open_gripper":
            tarjq = sim_node.getArmTargetPose(tarjq, task_command[0])
            if not tarjq.any():
                success = False
                return success

            state_cnt = 0
            while True:
                for i in range(6):
                    action[i] = sim_node.step_func(
                        action[i],
                        tarjq[i],
                        0.7
                        * sim_node.config.decimation
                        * sim_node.mj_model.opt.timestep,
                    )
                action[6] = sim_node.step_func(
                    action[6],
                    tarjq[6],
                    2.5 * sim_node.config.decimation * sim_node.mj_model.opt.timestep,
                )

                obs, pri_obs, rew, ter, info = sim_node.step(action)

                obs_lst.append(obs)
                act_lst.append(action.tolist())

                state_cnt += 1

                if (
                    state_cnt
                    * sim_node.config.decimation
                    * sim_node.mj_model.opt.timestep
                    > 0.9
                ):
                    print("Gripper is closed.")
                    if task_command[0] == "close_gripper":
                        sim_node.robot_state = (
                            "holding",
                            task_plan[task_command_i - 1][1],
                        )
                    else:
                        sim_node.robot_state = ("holding", None)
                    break

        else:
            if task_command[0] == "move_to":
                tarjq = sim_node.getArmTargetPose(
                    tarjq, task_command[0], task_command[1]
                )
            elif task_command[0] == "approach":
                tarjq = sim_node.getArmTargetPose(
                    tarjq, task_command[0], task_command[1], task_command[2]
                )

            if not tarjq.any():
                success = False
                return success

            if motion_planner:
                solu_list = motion_planner.get_motion_plan(obs["jq"][:6], tarjq[:6])
                if solu_list:
                    for solu in solu_list[1:]:
                        # solu = solu + [tarjq[6]]
                        # print("solu ", solu)
                        while not np.allclose(
                            obs["jq"][:6], solu[:6], atol=CONTROL_PRECISION
                        ):
                            for i in range(6):
                                action[i] = sim_node.step_func(
                                    action[i],
                                    solu[i],
                                    0.7
                                    * sim_node.config.decimation
                                    * sim_node.mj_model.opt.timestep,
                                )
                            # action[6] = sim_node.step_func(action[6], solu[6], 2.5 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                            # print("action ", action)
                            obs, pri_obs, rew, ter, info = sim_node.step(action)

                            obs_lst.append(obs)
                            act_lst.append(action.tolist())
                else:
                    success = False
                    return success

            else:
                while not np.allclose(obs["jq"][:6], tarjq[:6], atol=CONTROL_PRECISION):
                    for i in range(6):
                        action[i] = sim_node.step_func(
                            action[i],
                            tarjq[i],
                            0.7
                            * sim_node.config.decimation
                            * sim_node.mj_model.opt.timestep,
                        )
                    # action[6] = sim_node.step_func(action[6], tarjq[6], 2.5 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

                    obs, pri_obs, rew, ter, info = sim_node.step(action)

                    obs_lst.append(obs)
                    act_lst.append(action.tolist())

            # if np.allclose(obs["jq"][:6], tarjq[:6], atol=CONTROL_PRECISION):
            #    print("Coffeecup_white is picked up.")

    return success


def main(input_arguments):
    parser = argparse.ArgumentParser(description="Specify TAMP parameters.")
    parser.add_argument(
        "task_instances",
        type=str,
        nargs="+",
        help="Specify a task instance (str). \
                              Valid choices: plate_coffeecup, stack_1block1bowl)",
    )
    args = parser.parse_args(input_arguments)

    save_dir = join(motion_planning_dir, "results/")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    # setup logger
    logger = logging.getLogger()
    setup_global_logger(
        logger,
        file=save_dir
        + "/main_"
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        + ".log",
    )

    sim_node = AirbotPlaySimNode(args.task_instances[0])
    env_description = sim_node.getEnvDescription()
    print(env_description)
    motion_planner = AirbotPlayMotionPlanner(sim_node)
    llm_task_planner = LLMTaskPlanner(args.task_instances[0], env_description)

    task_plan = llm_task_planner.get_task_plan()
    success = executeTaskPlan(sim_node, task_plan, motion_planner)

    return


def benchmark():
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
