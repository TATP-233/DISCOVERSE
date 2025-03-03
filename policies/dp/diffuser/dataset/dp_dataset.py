from typing import Dict
import torch
import numpy as np
import copy
from .base_dataset import BaseDataset
from .replay_buffer import ReplayBuffer
from .dataset_utils import (
    SequenceSampler,
    MinMaxNormalizer,
    ImageNormalizer,
    dict_apply,
)


class DPDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        obs_keys=["image0", "image1", "jq"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys + ["action"]
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = {"obs": {}, "action": None}
        # 处理所有观测值的normalizer
        for key in self.replay_buffer.keys():
            if key == "action":
                normalizer["action"] = MinMaxNormalizer(self.replay_buffer[key][:])
            else:
                if key.startswith("image"):
                    normalizer["obs"][key] = ImageNormalizer()
                else:
                    normalizer["obs"][key] = MinMaxNormalizer(
                        self.replay_buffer[key][..., :]
                    )
        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {"obs": {}, "action": None}
        # 处理所有观测值
        for key in sample.keys():
            if key == "action":
                action = sample[key].astype(np.float32)
                data["action"] = self.normalizer["action"].normalize(action)
            else:
                if key.startswith("image"):
                    img = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255
                    data["obs"][key] = self.normalizer["obs"][key].normalize(img)
                else:
                    obs = sample[key][:, :].astype(np.float32)
                    data["obs"][key] = self.normalizer["obs"][key].normalize(obs)

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data
