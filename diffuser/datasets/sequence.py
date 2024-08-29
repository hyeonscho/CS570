from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple("Batch", "trajectories conditions returns")
Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-expert-v2",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        discount=0.99,
        returns_scale=1000,
        include_returns=False,
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

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
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch


class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-expert-v2",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=50000,
        termination_penalty=0,
        use_padding=True,
        discount=0.99,
        returns_scale=1000,
        include_returns=False,
        data_file=None,
        stitch=False,
        task_data=False,
        jump=1,
        aug_data_file=None,
    ):
        env_name = env
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        self.jump = jump

        if data_file is not None and not stitch:
            if "AntMaze_UMaze-v4" in env_name:
                dataset_name = "antmaze-umaze-v0"
            elif "AntMaze_Medium-v4" in env_name:
                dataset_name = "antmaze-medium-v0"
            elif "AntMaze_Large-v4" in env_name:
                dataset_name = "antmaze-large-v0"
            elif "PointMaze_UMaze-v3" in env_name:
                dataset_name = "pointmaze-umaze-v1"
            elif "PointMaze_Medium-v3" in env_name:
                dataset_name = "pointmaze-medium-v1"
            elif "PointMaze_Large-v3" in env_name:
                dataset_name = "pointmaze-large-v1"
            dataset = dataset_name + ".pkl"
            data_file = data_file + dataset

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        if not stitch:
            itr = sequence_dataset(
                env, self.preprocess_fn, data_file=data_file, task_data=task_data
            )

            for i, episode in enumerate(itr):
                fields.add_path(episode)

        if aug_data_file is not None:
            aug_iter = sequence_dataset(
                env, self.preprocess_fn, aug_data_file, task_data=False
            )
            for i, episode in enumerate(aug_iter):
                fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

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
            max_start = min(path_length, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start + 1):
                end = start + horizon
                indices.append((i, start, end))

        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end][
            :: self.jump
        ]
        actions = self.fields.normed_actions[path_ind, start:end][:: self.jump]
        traj_dim = self.action_dim + self.observation_dim

        t_step = 1
        horizon = self.horizon // self.jump
        conditions = np.ones((horizon, self.observation_dim * 2)).astype(np.float32)
        conditions[:, self.observation_dim :] = 0
        conditions[:t_step, : self.observation_dim] = 0
        conditions[:t_step, self.observation_dim :] = observations[:t_step]

        import random

        start_index = random.randint(1, horizon - 1)
        end_index = random.randint(start_index, horizon - 1)

        # [0, t_step] and [start_index, end_index] is masked
        conditions[start_index : end_index + 1, : self.observation_dim] = 0
        conditions[start_index : end_index + 1, self.observation_dim :] = observations[
            start_index : end_index + 1
        ]
        trajectories = np.concatenate(
            [actions, observations], axis=-1
        )  # shape is (100, 14)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
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
