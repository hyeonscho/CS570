# TODO
from collections import namedtuple
import numpy as np
import torch
import pdb
import os

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from math import pi
import h5py
from tqdm import tqdm
from d4rl.pointmaze import maze_model
from diffuser.utils.debug import debug
import re

Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")
LevelBatch = namedtuple("LevelBatch", "trajectories conditions levels")


class SequenceDatasetHMDMultiscale(torch.utils.data.Dataset):
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
        jumps=[1],
        jump_action=False,
        short_seq_len=1,
        make_multi_indices=True,
        use_stitched_data=False,
        use_short_data=False,
        max_round=-1,
        stitched_method="linear",
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.short_seq_len = short_seq_len
        load_path = None
        itr = sequence_dataset(env, self.preprocess_fn, load_path=load_path)
        self.jumps = jumps
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
        # in self.jumps -> there should be multiple jumps=1 -> each one for different level
        # They are splitted HL and LL -> LL is only J1
        # self.levels_per_jumps = {}
        # for j in self.jumps:
        #     if j > 1:
        #         self.levels_per_jumps[j] = 
        
        # self.jumps = [1, 1, 1, 10, 15, 20]
        _jumps = list(set(self.jumps))
        assert _jumps[0] == 1 and _jumps[1] > 1
        self.jumps_wo_ll = _jumps[1:]
        self.bins_lengths = [j*self.short_seq_len for j in self.jumps]
        if make_multi_indices:
            self.indices = [
                self.make_indices(fields.path_lengths, h) for h in self.bins_lengths
            ]
            print(f"bins_lengths: {self.bins_lengths}, indices: {len(self.indices)}")
        else:
            self.indices = self.make_indices(fields.path_lengths, self.horizon)

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

    def get_conditions(self, observations, end=None):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices[0]) # the one that had the largest amount of trajectories because it has smallest jump and horizon 

    def __getitem__(self, idx, eps=1e-4):
        ok = False
        # from diffuser.utils.debug import debug
        # debug()
        random_level = np.random.randint(0, len(self.jumps))
        jump_idx = random_level #self.params.get('jump_idx', None) or random_level
        # if self.jumps[jump_idx] == 1:
        #     jump_idx = 0
        # else:
        #     jump_idx -= (len(self.jumps)-len(self.jumps_wo_ll))+1

        indx_len = len(self.indices[jump_idx])
        idx = idx % indx_len
        
        path_ind, start, end = self.indices[jump_idx][idx]
        self.jump = self.jumps[jump_idx]
        
        random_condition_masking = None
        if self.jump == 1:
            random_condition_masking = np.random.randint(0, len(self.jumps_wo_ll))
            random_condition_masking = int(self.jumps_wo_ll[random_condition_masking])
            
            
        observations = self.fields.normed_observations[path_ind, start:end][
            :: self.jump
        ]
        levels = np.eye(len(self.jumps), dtype=observations.dtype)[jump_idx] # one-hot encoding

        conditions = self.get_conditions(observations, end=random_condition_masking)
        reshape = lambda x: x.reshape([x.shape[0] // self.jump, self.jump] + list(x.shape[1:]))
        if self.jump_action == "none":
            trajectories = observations
        else:
            # raise NotImplementedError("jump_action != none")
            # debug()
            actions = self.fields.normed_actions[path_ind, start:end]
            actions = reshape(actions)[:, :self.jump_action]
            actions = actions.reshape(actions.shape[0], -1)
            # From HD:
            # padd with all the actions in the jump sequence
            # actions = self.fields.normed_actions[path_ind, start:end].reshape(-1, self.jump * self.action_dim)
            trajectories = np.concatenate([actions, observations], axis=-1)
        # batch = Batch(trajectories, conditions)
        batch = LevelBatch(trajectories, conditions, levels)
        return batch

# That's exactly same as the class in sequence_hmd.py
class GoalDatasetHMDMultiscale(SequenceDatasetHMDMultiscale):
    def get_conditions(self, observations, end=None):
        """
        condition on both the current observation and the last observation in the plan
        """
        end = self.short_seq_len-1
        return {
            0: observations[0],
            end: observations[end],
        }

class GoalDatasetHMDMultiscale2(SequenceDatasetHMDMultiscale):
    def get_conditions(self, observations, end=None):
        """
        condition on both the current observation and the last observation in the plan
        """
        end = self.short_seq_len-1 if end is None else end
        return {
            0: observations[0],
            'key': np.full_like(observations[0], end),
            'value': observations[end],
        }



class RandomGoalDatasetHMDMultiscale(SequenceDatasetHMDMultiscale):
    def get_conditions(self, observations, end=None):
        """
        condition on both the current observation and the last observation in the plan
        """
        goal_idx = np.random.choice(np.arange(self.short_seq_len-5, self.short_seq_len-1))
        return {
            0: observations[0],
            goal_idx: observations[goal_idx],
        }


class ValueDatasetHMDMultiscale(SequenceDatasetHMDMultiscale):
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
