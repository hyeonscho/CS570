from collections import namedtuple
import numpy as np
import torch
import os

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
import re

Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        jump=1,
        jump_action=False,
        use_stitched_data=False,
        use_short_data=False,
        max_round=-1,
        stitched_method="linear", # "linear"
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        load_path = None
        self.jump = jump
        self.jump_action = jump_action
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        if use_stitched_data:
            parent_data_file = "/root/diffuser_chain_hd/data/"
            if use_short_data:
                # I need to postprocess it first
                print("Using the short dataset")
                data_file = os.path.join(parent_data_file, f"{self.env_name}-base-overlapped.pkl")
                _preprocess_fn = get_preprocess_fn(['postprocess_base'], env)
                itr = sequence_dataset(env, _preprocess_fn, load_path=data_file, use_final_timestep=False)
                for i, episode in enumerate(itr):
                    fields.add_path(episode)
            print("Using the stitched dataset")
            data_file = os.path.join(parent_data_file, f"{self.env_name}-{stitched_method}-round_{max_round}-postprocess.pkl")
            pattern = r"round_(\d+)"
            match = re.search(pattern, data_file)
            last_round_num = int(match.group(1))
            _preprocess_fn = get_preprocess_fn(['postprocess_stitched'], env)
            for r in range(1, last_round_num + 1):
                aug_data_file_round_r = data_file.replace(f"round_{last_round_num}", f"round_{r}")
                print(f"Loading {aug_data_file_round_r}")
                aug_iter = sequence_dataset(env, _preprocess_fn, aug_data_file_round_r, use_final_timestep=False)
                for i, episode in enumerate(aug_iter):
                    fields.add_path(episode)
        else:
            print("Using the original dataset")
            itr = sequence_dataset(env, self.preprocess_fn, load_path=load_path) # that is the original dataset
            for i, episode in enumerate(itr):
                fields.add_path(episode)
        fields.finalize()
        self.fields = fields

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def normalize(self, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def make_indices(self, path_lengths, horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))

        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        ok = False
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end][
            :: self.jump
        ]
        actions = self.fields.normed_actions[path_ind, start:end].reshape(
            -1, self.jump * self.action_dim
        )

        conditions = self.get_conditions(observations)
        if self.jump_action == "none":
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):
    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon // self.jump : observations[-1],
        }


class RandomGoalDataset(SequenceDataset):
    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        goal_idx = np.random.choice(np.arange(15, self.horizon // self.jump))
        return {
            0: observations[0],
            goal_idx: observations[goal_idx],
        }


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields["rewards"][path_ind, start:]
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
