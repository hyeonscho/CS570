from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import urllib
import os
import ogbench
from .normalization import DatasetNormalizer
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_RENDERER"] = "egl"
from collections import namedtuple
Batch = namedtuple("Batch", "trajectories conditions")

class OGCubeOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            env='cube-single-play-v0',
            horizon=200,
            normalizer='LimitsNormalizer',
            preprocess_fns=[],
            max_path_length=200,   
            max_n_episodes=10000,
            termination_penalty=0,
            use_padding=True,
            jump=1,
            jump_action=False,
            use_stitched_data=False,
            use_short_data=False,
            max_round=-1,
            stitched_method='linear',
            split="training",
            only_start_condition = False,
        ): #cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.env = env
        self.horizon = horizon
        self.normalizer = normalizer
        self.preprocess_fns = preprocess_fns
        self.max_path_length = max_path_length
        self.max_n_episodes = max_n_episodes
        self.termination_penalty = termination_penalty
        self.use_padding = use_padding
        self.jump = jump
        self.jump_action = jump_action
        self.use_stitched_data = use_stitched_data
        self.use_short_data = use_short_data
        self.max_round = max_round
        self.stitched_method = stitched_method
        self.split = split
        self.only_start_condition = only_start_condition
        # self.cfg = cfg
        #self.save_dir = cfg.save_dir # Using default save_dir, "~/.ogbench/data"
        # self.env_id = cfg.env_id
        self.dataset_name = env
        self.n_frames = self.horizon
        self.gamma = 1.0
        self.split = split
        #Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        # Use [position, velocity] as observation
        obs = self.dataset['observations'] # (num_data, 19 + num_cubes*9)
        num_cubes = (obs.shape[1] - 19) // 9 # 1,2,3,4
        obs = np.concatenate([obs[:,19+i*9:19+i*9+3] for i in range(num_cubes)], axis=-1) # cube positions only
        self.dataset['observations'] = obs

        sample_length = 1001 # ?

        assert self.n_frames <= sample_length, f"Episode length {self.n_frames} is greater than sample length {sample_length}"

        # Dataset Statistics
        print(f"Dataset: {self.dataset_name}")
        print(f"Total samples: {len(self.dataset['observations'])}, Subtrajectory length: {sample_length}")
        obs_mean = np.mean(self.dataset["observations"], axis=0)
        self.observation_dim = obs_mean.shape[0]
        obs_std = np.std(self.dataset["observations"], axis=0)
        print(f"Observation shape: {self.dataset['observations'].shape}")
        print(f"Observation mean: {obs_mean}")
        print(f"Observation std:  {obs_std}")
        act_mean = np.mean(self.dataset["actions"], axis=0)
        self.action_dim = act_mean.shape[0]
        act_std = np.std(self.dataset["actions"], axis=0)
        print(f"Action shape: {self.dataset['actions'].shape}")
        print(f"Action mean: {act_mean}")
        print(f"Action std:  {act_std}")

        # Dataset Reshaping
        raw_observations = np.reshape(self.dataset["observations"], (-1, sample_length, self.dataset["observations"].shape[-1]))
        raw_actions = np.reshape(self.dataset["actions"], (-1, sample_length, self.dataset["actions"].shape[-1]))
        raw_terminals = np.zeros((raw_observations.shape[0], sample_length)) # This will not be used in training and validation
        raw_terminals[:, -1] = 1
        raw_rewards = np.copy(raw_terminals)
        raw_values = self.compute_value(raw_rewards) * (1 - self.gamma) * 4 - 1

        # Dataset Preprocessing (Collecting episode_len trajectories in sliding window manner)
        self.observations, self.actions, self.rewards, self.values = [], [], [], []
        for i in range(raw_observations.shape[0]):
            for j in range(sample_length - self.n_frames):
                self.observations.append(raw_observations[i, j:j+self.n_frames])
                self.actions.append(raw_actions[i, j:j+self.n_frames])
                self.rewards.append(raw_rewards[i, j:j+self.n_frames])
                self.values.append(raw_values[i, j:j+self.n_frames])
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values)

        # dataset normalization
        # breakpoint()
        self.normalizer = DatasetNormalizer({'observations':self.observations.reshape(-1, self.observations.shape[-1]), 'actions':self.actions.reshape(-1, self.actions.shape[-1])}, normalizer)
        self.normalized_obs = self.normalizer.normalize(self.observations.reshape(-1, self.observations.shape[-1]), 'observations').reshape(-1, self.n_frames, self.observations.shape[-1])
        self.normalized_act = self.normalizer.normalize(self.actions.reshape(-1, self.actions.shape[-1]), 'actions').reshape(-1, self.n_frames, self.actions.shape[-1])
        # Preprocessed Dataset Statistics
        print(f"Preprocessed Dataset Statistics")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        print(f"Reward shape: {self.rewards.shape}")
        print(f"Value shape: {self.values.shape}")
        self.total_samples = len(self.observations)

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(reward.shape[1] - 2, -1, -1):
            value[:, i] += self.gamma * value[:, i + 1]
        return value

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        observations = torch.from_numpy(self.normalized_obs[idx])[::self.jump].float() # (episode_len, obs_dim)
        actions = torch.from_numpy(self.normalized_act[idx]).reshape(-1, self.jump*self.action_dim).float()
        conditions = self.get_conditions(observations, self.only_start_condition)
        if self.jump_action == "none":
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        # action = torch.from_numpy(self.actions[idx]).float() # (episode_len, act_dim)
        # reward = torch.from_numpy(self.rewards[idx]).float() # (episode_len,)
        # value = torch.from_numpy(self.values[idx]).float() # (episode_len,)
        # done = np.zeros(len(observation), dtype=bool)
        # done[-1] = True
        # nonterminal = torch.from_numpy(~done)
        batch = Batch(trajectories, conditions)

        return batch

    def get_dataset(self):
        _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            self.dataset_name,
            #self.save_dir, # Using default save_dir, "~/.ogbench/data"
            compact_dataset=True,
        )
        if self.split == "training":
            return train_dataset
        else:
            return val_dataset
        
    def get_conditions(self, observations, only_start_condition):
        """
        condition on both the current observation and the last observation in the plan
        """
        if only_start_condition:
            return {
                0: observations[0],
            }

        return {
            0: observations[0],
            self.horizon // self.jump - 1: observations[-1],
        }
