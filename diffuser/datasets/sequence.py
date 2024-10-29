from collections import namedtuple
from typing import NamedTuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

# RewardBatch = namedtuple("Batch", "trajectories conditions returns")
# Batch = namedtuple("Batch", "trajectories conditions")
# ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")
# LevelBatch = namedtuple("LevelBatch", "trajectories conditions levels")
# LevelRewardBatch = namedtuple("LevelBatch", "trajectories conditions levels returns")


class RewardBatch(NamedTuple):
    trajectories: np.float32
    conditions: np.float32
    returns: np.float32


class Batch(NamedTuple):
    trajectories: np.float32
    conditions: np.float32


class ValueBatch(NamedTuple):
    trajectories: np.float32
    conditions: np.float32
    values: np.float32


class LevelBatch(NamedTuple):
    trajectories: np.float32
    conditions: np.float32
    levels: np.float32


class LevelRewardBatch(NamedTuple):
    trajectories: np.float32
    conditions: np.float32
    levels: np.float32
    returns: np.float32


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
        max_n_episodes=800000,
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
        segment_return=False,
        jumps=[],
        task_len=None,
        act_pad=False,
        cum_rew=False,
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
        self.jumps = jumps
        self.level_dim = len(self.jumps)
        self.segment_return = segment_return
        self.act_pad = act_pad
        self.cum_rew = cum_rew
        if self.jumps:
            self.segmt_len = int(np.ceil(self.horizon / self.jumps[-1]))
        else:
            self.segmt_len = horizon

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
                env,
                self.preprocess_fn,
                data_file=data_file,
                task_data=task_data,
                task_len=task_len,
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
        horizon = int(np.ceil(self.horizon / self.jump))
        jump = self.jump

        observations = self.fields.normed_observations[path_ind, start:end][::jump]
        if self.act_pad:
            pad_actions = self.fields.normed_actions[
                path_ind, start : start + jump * horizon
            ].reshape(horizon, jump * self.action_dim)
            observations = np.concatenate([observations, pad_actions], axis=-1)
        actions = self.fields.normed_actions[path_ind, start:end][::jump]
        traj_dim = self.action_dim + self.observation_dim

        t_step = 1
        obs_dim = observations.shape[-1]
        conditions = np.ones((horizon, obs_dim * 2)).astype(np.float32)
        conditions[:, obs_dim:] = 0
        # conditions[:, obs_dim : obs_dim + self.observation_dim] = 0
        conditions[:t_step, : self.observation_dim] = 0
        conditions[:t_step, obs_dim : obs_dim + self.observation_dim] = observations[
            :t_step, : self.observation_dim
        ]

        import random

        start_index = random.randint(1, horizon - 1)
        end_index = random.randint(start_index, horizon - 1)

        # todo: mask only for stitcher
        # [0, t_step] and [start_index, end_index] is masked
        # conditions[start_index : end_index + 1, : self.observation_dim] = 0
        # conditions[start_index : end_index + 1, self.observation_dim :] = observations[
        #     start_index : end_index + 1
        # ]
        trajectories = np.concatenate(
            [actions, observations], axis=-1
        )  # shape is (100, 14)

        if self.include_returns:
            if self.segment_return:
                path_len = self.fields.path_lengths[path_ind]
                rewards = self.fields.rewards[path_ind, start:end]
            else:
                rewards = self.fields.rewards[path_ind, start:]
            if self.cum_rew:
                rewards = np.cumsum(rewards, axis=0)
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch


class CondLLSequenceDataset(CondSequenceDataset):
    def __init__(
        self,
        env="hopper-medium-expert-v2",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=800000,
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
        segment_return=False,
        jumps=[],
        task_len=None,
        act_pad=False,
        cum_rew=False,
    ):
        super().__init__(
            env,
            horizon,
            normalizer,
            preprocess_fns,
            max_path_length,
            max_n_episodes,
            termination_penalty,
            use_padding,
            discount,
            returns_scale,
            include_returns,
            data_file,
            stitch,
            task_data,
            jump,
            aug_data_file,
            segment_return,
            jumps,
            task_len,
            act_pad,
            cum_rew,
        )

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        horizon = int(np.ceil(self.horizon / self.jump))
        jump = self.jump

        observations = self.fields.normed_observations[path_ind, start:end][::jump]
        actions = self.fields.normed_actions[path_ind, start:end][::jump]
        traj_dim = self.action_dim + self.observation_dim

        t_step = 1
        conditions = np.ones((horizon, self.observation_dim * 2)).astype(np.float32)
        conditions[:, self.observation_dim :] = 0
        conditions[:t_step, : self.observation_dim] = 0
        conditions[:t_step, self.observation_dim :] = observations[:t_step]

        conditions[-1:, : self.observation_dim] = 0
        conditions[-1:, self.observation_dim :] = observations[-1:]

        import random

        start_index = random.randint(1, horizon - 1)
        end_index = random.randint(start_index, horizon - 1)

        # todo: mask only for stitcher
        # [0, t_step] and [start_index, end_index] is masked
        # conditions[start_index : end_index + 1, : self.observation_dim] = 0
        # conditions[start_index : end_index + 1, self.observation_dim :] = observations[
        #     start_index : end_index + 1
        # ]
        trajectories = np.concatenate(
            [actions, observations], axis=-1
        )  # shape is (100, 14)

        if self.include_returns:
            if self.segment_return:
                path_len = self.fields.path_lengths[path_ind]
                rewards = self.fields.rewards[path_ind, start:end]
            else:
                rewards = self.fields.rewards[path_ind, start:]
            if self.cum_rew:
                rewards = np.cumsum(rewards, axis=-1)
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch


class CondCLSequenceDataset(CondSequenceDataset):
    def __init__(
        self,
        env="hopper-medium-expert-v2",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=80000,
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
        segment_return=False,
        jumps=[],
        task_len=None,
        act_pad=False,
        cum_rew=False,
    ):
        super().__init__(
            env,
            horizon,
            normalizer,
            preprocess_fns,
            max_path_length,
            max_n_episodes,
            termination_penalty,
            use_padding,
            discount,
            returns_scale,
            include_returns,
            data_file,
            stitch,
            task_data,
            jump,
            aug_data_file,
            segment_return,
            jumps,
            task_len,
            act_pad,
            cum_rew,
        )
        horizons = [
            1 + j if 1 + j > self.segmt_len else self.segmt_len for j in jumps[1:]
        ]
        horizons = horizons + [horizon]
        # self.indices = [
        #     self.make_indices(self.fields.path_lengths, h) for h in horizons
        # ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=0.0001):
        num_levels = len(self.jumps)
        random_level = np.random.randint(0, len(self.jumps))

        # indx_len = len(self.indices[random_level])
        # idx = idx % indx_len
        # path_ind, start, end = self.indices[random_level][idx]

        path_ind, start, end = self.indices[idx]
        jump = self.jumps[random_level]
        horizon = self.segmt_len

        observations = self.fields.normed_observations[path_ind, start:end][::jump]
        actions = self.fields.normed_actions[path_ind, start:end][::jump]

        random_idx = 0
        if len(observations) > horizon:
            random_idx = np.random.randint(0, len(observations) - horizon)
            observations = observations[random_idx : random_idx + horizon]
            actions = actions[random_idx : random_idx + horizon]

        levels = np.eye(len(self.jumps), dtype=observations.dtype)[random_level]
        levels = np.repeat(levels[None], horizon, axis=0)

        traj_dim = self.action_dim + self.observation_dim + self.level_dim

        t_step = 1
        conditions = np.ones((horizon, self.observation_dim * 2)).astype(np.float32)
        conditions[:, self.observation_dim :] = 0
        conditions[:t_step, : self.observation_dim] = 0
        conditions[:t_step, self.observation_dim :] = observations[:t_step]

        import random

        # if random_level == 0 and self.jumps[1] + 1 < horizon:
        end_index = horizon
        if random_level < num_levels - 1:
            level_horizon = int(
                np.ceil((self.jumps[random_level + 1] + 1) / self.jumps[random_level])
            )
            if level_horizon < horizon:
                end_index = level_horizon

        start_index = random.randint(1, end_index)
        end_index = random.randint(start_index, end_index)

        # todo: lower levels where level_horizon > horizon
        if random_level < num_levels - 1:
            if level_horizon < horizon:
                # conditions[start_index : end_index + 1, : self.observation_dim] = 0
                # conditions[start_index : end_index + 1, self.observation_dim :] = (
                #     observations[start_index : end_index + 1]
                # )
                conditions[level_horizon - 1 :, : self.observation_dim] = 0
                conditions[level_horizon - 1 :, self.observation_dim :] = observations[
                    level_horizon - 1 : level_horizon
                ]
            else:
                conditions[-1:, : self.observation_dim] = 0
                conditions[-1:, self.observation_dim :] = observations[-1:]

        # else:
        # [0, t_step] and [start_index, end_index] is masked
        # conditions[start_index : end_index + 1, : self.observation_dim] = 0
        # conditions[start_index : end_index + 1, self.observation_dim :] = (
        #     observations[start_index : end_index + 1]
        # )
        trajectories = np.concatenate(
            [actions, observations], axis=-1
        )  # shape is (100, 14)

        if self.include_returns:
            if self.segment_return:
                rewards = self.fields.rewards[path_ind, start:end]
            else:
                rewards = self.fields.rewards[path_ind, start + random_idx :]
            if self.cum_rew:
                rewards = np.cumsum(rewards, axis=-1)
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = LevelRewardBatch(trajectories, conditions, levels, returns)
        else:
            batch = LevelBatch(trajectories, conditions, levels)

        return batch


class CondCLSequenceDatasetV2(CondCLSequenceDataset):
    def __init__(
        self,
        env="hopper-medium-expert-v2",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=80000,
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
        segment_return=False,
        jumps=[],
        task_len=None,
        act_pad=False,
        cum_rew=False,
    ):
        super().__init__(
            env,
            horizon,
            normalizer,
            preprocess_fns,
            max_path_length,
            max_n_episodes,
            termination_penalty,
            use_padding,
            discount,
            returns_scale,
            include_returns,
            data_file,
            stitch,
            task_data,
            jump,
            aug_data_file,
            segment_return,
            jumps,
            task_len,
            act_pad,
            cum_rew,
        )
        horizons = [
            1 + j if 1 + j > self.segmt_len else self.segmt_len for j in jumps[1:]
        ]
        horizons = horizons + [horizon]
        # self.indices = [
        #     self.make_indices(self.fields.path_lengths, h) for h in horizons
        # ]

    def __getitem__(self, idx, eps=0.0001):
        num_levels = len(self.jumps)
        random_level = np.random.randint(0, len(self.jumps))

        # indx_len = len(self.indices[random_level])
        # idx = idx % indx_len
        # path_ind, start, end = self.indices[random_level][idx]

        path_ind, start, end = self.indices[idx]
        jump = self.jumps[random_level]
        horizon = self.segmt_len

        observations = self.fields.normed_observations[path_ind, start:end][::jump]
        actions = self.fields.normed_actions[path_ind, start:end][::jump]
        if self.act_pad:
            observations = np.concatenate([observations, actions], axis=-1)

        random_idx = 0
        if len(observations) > horizon:
            random_idx = np.random.randint(0, len(observations) - horizon)
            observations = observations[random_idx : random_idx + horizon]
            actions = actions[random_idx : random_idx + horizon]

        levels = np.eye(len(self.jumps), dtype=observations.dtype)[random_level]
        levels = np.repeat(levels[None], horizon, axis=0)

        traj_dim = self.action_dim + self.observation_dim + self.level_dim

        t_step = 1
        obs_dim = observations.shape[-1]
        conditions = np.ones((horizon, obs_dim * 2)).astype(np.float32)
        conditions[:, obs_dim:] = 0
        conditions[:t_step, : self.observation_dim] = 0
        conditions[:t_step, obs_dim : obs_dim + self.observation_dim] = observations[
            :t_step, : self.observation_dim
        ]

        import random

        # if random_level == 0 and self.jumps[1] + 1 < horizon:
        end_index = horizon
        if random_level < num_levels - 1:
            level_horizon = int(
                np.ceil((self.jumps[random_level + 1] + 1) / self.jumps[random_level])
            )
            if level_horizon < horizon:
                end_index = level_horizon

        start_index = random.randint(1, end_index)
        end_index = random.randint(start_index, end_index)

        # todo: lower levels where level_horizon > horizon
        if random_level < num_levels - 1:
            if level_horizon < horizon:
                # conditions[start_index : end_index + 1, : self.observation_dim] = 0
                # conditions[start_index : end_index + 1, self.observation_dim :] = (
                #     observations[start_index : end_index + 1]
                # )
                # a. mask the sub-goal
                conditions[
                    level_horizon - 1 : level_horizon, : self.observation_dim
                ] = 0
                conditions[
                    level_horizon - 1 : level_horizon,
                    obs_dim : obs_dim + self.observation_dim,
                ] = observations[
                    level_horizon - 1 : level_horizon, : self.observation_dim
                ]
                # b. mask random following states, could mask all or none
                start_index = random.randint(level_horizon, horizon)
                end_index = random.randint(start_index, horizon)
                conditions[start_index:end_index, : self.observation_dim] = 0
                conditions[
                    start_index:end_index, obs_dim : obs_dim + self.observation_dim :
                ] = observations[start_index:end_index, : self.observation_dim]
            else:
                conditions[-1:, : self.observation_dim] = 0
                conditions[-1:, obs_dim : obs_dim + self.observation_dim] = (
                    observations[-1:, : self.observation_dim]
                )

        # else:
        # [0, t_step] and [start_index, end_index] is masked
        # conditions[start_index : end_index + 1, : self.observation_dim] = 0
        # conditions[start_index : end_index + 1, self.observation_dim :] = (
        #     observations[start_index : end_index + 1]
        # )
        trajectories = np.concatenate(
            [actions, observations], axis=-1
        )  # shape is (100, 14)

        if self.include_returns:
            if self.segment_return:
                rewards = self.fields.rewards[path_ind, start:end]
            else:
                rewards = self.fields.rewards[path_ind, start + random_idx :]
            if self.cum_rew:
                rewards = np.cumsum(rewards, axis=-1)
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = LevelRewardBatch(trajectories, conditions, levels, returns)
        else:
            batch = LevelBatch(trajectories, conditions, levels)

        return batch


class CondCLWithShortSequenceDataset(CondSequenceDataset):
    def __init__(
        self,
        env="hopper-medium-expert-v2",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=80000,
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
        segment_return=False,
        jumps=[],
    ):
        super().__init__(
            env,
            horizon,
            normalizer,
            preprocess_fns,
            max_path_length,
            max_n_episodes,
            termination_penalty,
            use_padding,
            discount,
            returns_scale,
            include_returns,
            data_file,
            stitch,
            task_data,
            jump,
            aug_data_file,
            segment_return,
            jumps,
        )
        horizons = [
            1 + j if 1 + j > self.segmt_len else self.segmt_len for j in jumps[1:]
        ]
        horizons = horizons + [horizon]
        self.indices = [
            self.make_indices(self.fields.path_lengths, h) for h in horizons
        ]

    def __len__(self):
        return len(self.indices[0])

    def __getitem__(self, idx, eps=0.0001):
        random_level = np.random.randint(0, len(self.jumps))

        indx_len = len(self.indices[random_level])
        idx = idx % indx_len

        path_ind, start, end = self.indices[random_level][idx]
        jump = self.jumps[random_level]
        horizon = self.segmt_len

        observations = self.fields.normed_observations[path_ind, start:end][::jump]
        actions = self.fields.normed_actions[path_ind, start:end][::jump]
        levels = np.eye(len(self.jumps), dtype=observations.dtype)[random_level]
        levels = np.repeat(levels[None], horizon, axis=0)

        traj_dim = self.action_dim + self.observation_dim + self.level_dim

        t_step = 1
        conditions = np.ones((horizon, self.observation_dim * 2)).astype(np.float32)
        conditions[:, self.observation_dim :] = 0
        conditions[:t_step, : self.observation_dim] = 0
        conditions[:t_step, self.observation_dim :] = observations[:t_step]

        import random

        if random_level == 0 and self.jumps[1] + 1 < horizon:
            end_index = self.jumps[1]
        else:
            end_index = horizon

        start_index = random.randint(1, end_index)
        end_index = random.randint(start_index, end_index)

        if random_level == 0 and self.jumps[1] + 1 < horizon:
            conditions[start_index : end_index + 1, : self.observation_dim] = 0
            conditions[start_index : end_index + 1, self.observation_dim :] = (
                observations[start_index : end_index + 1]
            )
            conditions[self.jumps[1] :, : self.observation_dim] = 0
            conditions[self.jumps[1] :, self.observation_dim :] = observations[
                self.jumps[1] : self.jumps[1] + 1
            ]
        else:
            # [0, t_step] and [start_index, end_index] is masked
            conditions[start_index : end_index + 1, : self.observation_dim] = 0
            conditions[start_index : end_index + 1, self.observation_dim :] = (
                observations[start_index : end_index + 1]
            )
        trajectories = np.concatenate(
            [actions, observations], axis=-1
        )  # shape is (100, 14)

        if self.include_returns:
            if self.segment_return:
                rewards = self.fields.rewards[path_ind, start:end]
            else:
                rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = LevelRewardBatch(trajectories, conditions, levels, returns)
        else:
            batch = LevelBatch(trajectories, conditions, levels)

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
