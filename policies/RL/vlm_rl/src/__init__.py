"""
VLM-RL: Vision-Language Model guided Reinforcement Learning for DISCOVERSE

This module provides tools for:
1. Generating candidate keypoints from MuJoCo scenes
2. Using VLM to generate task constraints from images and instructions
3. Converting constraints to RL rewards
4. Training policies with PPO
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Automatically searches for .env in current and parent directories

from .keypoint_proposer import KeypointProposer
from .annotated_renderer import AnnotatedRenderer
from .constraint_generator import ConstraintGenerator, ConstraintLoader
from .keypoint_tracker import KeypointTracker
from .reward_adapter import ConstraintRewardAdapter
from .env import VLMRLEnv
from .standalone_env import StandaloneVLMRLEnv, make_standalone_env

__all__ = [
    "KeypointProposer",
    "AnnotatedRenderer",
    "ConstraintGenerator",
    "ConstraintLoader",
    "KeypointTracker",
    "ConstraintRewardAdapter",
    "VLMRLEnv",
    "StandaloneVLMRLEnv",
    "make_standalone_env",
]
