import shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import json

from os.path import abspath, dirname, join
import os
import sys

base_dir = os.getcwd()
sys.path.insert(0, base_dir)
motion_planning_dir = join(base_dir, "discoverse/motion_planning")
sys.path.insert(1, motion_planning_dir)
print(motion_planning_dir)

import copy
import mediapy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import multiprocessing as mp
from utils import load_txt

import abc
import httpx
from openai import OpenAI
from retry import retry

engine_choices = ["gpt-3.5-turbo", "gpt-4o-mini-2024-07-18", "gpt-4-0125-preview"]
DefaultConfigs = {
    "llm_params": {
        "engine": "gpt-4o-mini-2024-07-18",
        "key": "xxxx",
        "org": "xxxx",
    }
}


class LLMBase(abc.ABC):
    def __init__(self, engine, api_key, organization):
        # engine = "gpt-4-0125-preview" if use_gpt_4 else "gpt-3.5-turbo"
        self.llm_gpt = GPT_Chat(api_key, organization, engine)

    def prompt_llm(
        self, prompt: str, temperature: float = 0.0, force_json: bool = False
    ):
        # feed prompt to llm
        logger.info("\n" + "#" * 50)
        logger.info(f"Prompt:\n{prompt}")
        messages = [{"role": "user", "content": prompt}]

        conn_success, llm_output = self.llm_gpt.get_response(
            prompt=None,
            messages=messages,
            end_when_error=False,
            temperature=temperature,
            force_json=force_json,
        )
        if not conn_success:
            raise Exception("Fail to connect to the LLM")

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM output:\n{llm_output}")

        return llm_output


class LLMTaskPlanner(LLMBase):
    def __init__(self, task_instances, env_descriptions, cfg=DefaultConfigs):
        self.task_instances = task_instances
        self.env_descriptions = env_descriptions

        problem_description_dir = join(
            motion_planning_dir,
            "task_instances/task_descriptions/{}.txt".format(self.task_instances),
        )
        task_requirement_dir = join(motion_planning_dir, "params/task_requirements.txt")
        self.problem_descriptions = load_txt(problem_description_dir)
        self.task_requirements = load_txt(task_requirement_dir)

        self.llm_params = cfg["llm_params"]

        LLMBase.__init__(
            self,
            engine=self.llm_params["engine"],
            api_key=self.llm_params["key"],
            organization=self.llm_params["org"],
        )

    def prepare_planning_prompt(self):
        planning_prompt = "Problem descriptions: {}  \
                           Environment descriptions: {}  \
                           Task requirements: {} ".format(
            self.problem_descriptions, self.env_descriptions, self.task_requirements
        )

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM planning prompt:\n{planning_prompt}")

        return planning_prompt

    def llm_output_to_plan(self, planning_prompt):
        task_plan = []
        return task_plan

    def get_task_plan(self):
        task_plan = []
        planning_prompt = self.prepare_planning_prompt()

        llm_output = self.prompt_llm(prompt=planning_prompt, force_json=True)
        task_plan = json.loads(llm_output)["full_plan"]

        if self.task_instances == "plate_coffeecup":
            """
            task_plan = [
                ["move_to", "coffeecup_white"],
                ["close_gripper"],
                ["approach", "coffeecup_white", 0.14],
                ["approach", "plate_white", 0.12],
                ["approach", "plate_white", 0.083],
                ["open_gripper"],
                ["approach", "plate_white", 0.12],
                ["approach", "cup_lid", 0.14],
                ["approach", "cup_lid", 0.069],
                ["close_gripper"],
                ["approach", "cup_lid", 0.14],
                ["move_to", "plate_white"],
                ["open_gripper"],
                ["approach", "plate_white", 0.153]]
            """
            task_plan = [
                ["approach", "coffeecup_white", 0.1],
                ["move_to", "coffeecup_white"],
                ["close_gripper"],
                ["approach", "coffeecup_white", 0.1],
                ["approach", "plate_white", 0.1],
                ["move_to", "plate_white"],
                ["open_gripper"],
                ["approach", "plate_white", 0.1],
                ["approach", "cup_lid", 0.1],
                ["move_to", "cup_lid"],
                ["close_gripper"],
                ["approach", "cup_lid", 0.1],
                ["approach", "coffeecup_white", 0.1],
                ["move_to", "coffeecup_white"],
                ["open_gripper"],
                ["approach", "coffeecup_white", 0.1],
            ]

        elif self.task_instances == "stack_1block1bowl":

            task_plan = [
                ["approach", "block_green", 0.1],
                ["approach", "block_green", 0.025],
                ["close_gripper"],
                ["approach", "block_green", 0.1],
                ["approach", "bowl_pink", 0.14],
                ["approach", "bowl_pink", 0.07],
                ["open_gripper"],
                ["approach", "bowl_pink", 0.14],
            ]

        print("LLM task plan is ", task_plan)
        return task_plan


@retry(tries=5, delay=60)
def connect_openai(
    client,
    engine,
    messages,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    # stop,
    response_format,
):
    return client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
    )


class GPT_Chat:
    def __init__(
        self,
        api_key,
        organization,
        engine,
        stop=None,
        max_tokens=1000,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        # add key
        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            timeout=60,
            max_retries=5,
        )

    def get_response(
        self,
        prompt,
        messages=None,
        end_when_error=False,
        max_retry=2,
        temperature=0.0,
        force_json=False,
    ):
        conn_success, llm_output = False, ""
        if messages is not None:
            messages = messages
        else:
            messages = [{"role": "user", "content": prompt}]

        if force_json:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        n_retry = 0
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                logger.info("[INFO] connecting to the LLM ...")

                response = connect_openai(
                    client=self.client,
                    engine=self.engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.freq_penalty,
                    presence_penalty=self.presence_penalty,
                    response_format=response_format,
                )
                llm_output = response.choices[0].message.content
                conn_success = True
            except Exception as e:
                logger.info("[ERROR] LLM error: {}".format(e))
                if end_when_error:
                    break
        return conn_success, llm_output
