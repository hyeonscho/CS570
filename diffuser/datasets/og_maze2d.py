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

class OGMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    '''
    Offline RL dataset for 2D maze environments from OG-Bench.
    - Dataset: pointmaze-medium-navigate-v0
        - Total samples: 1000100, Subtrajectory length: 1001
        - Observation shape: (1000100, 2)
        - Observation mean: [1.0233182e+01  9.5702305e+00  1.5680445e-04 -3.4479308e-05]
        - Observation std:  [5.8341284  5.1942716  0.03949062 0.03930504]
        - Action shape: (1000100, 2)
        - Action mean: [0.00319824 -0.00255336]
        - Action std:  [0.705255  0.7001879]
    - Dataset: pointmaze-medium-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [9.5560627e+00  1.0342426e+01  5.5203458e-05 -3.1391672e-05]
        - Observation std:  [5.79129    5.055657   0.02651537 0.0255206]
        - Action shape: (1005000, 2)
        - Action mean: [-0.01176269  0.00561046]
        - Action std:  [0.7088544 0.7083117]
    - Dataset: pointmaze-large-navigate-v0
        - Total samples: 1000100, Subtrajectory length: 1001
        - Observation shape: (1000100, 2)
        - Observation mean: [1.6903608e+01  1.1005926e+01 -1.5858751e-04 -9.3383751e-05]
        - Observation std:  [10.277822    7.008117    0.03491852  0.03474263]
        - Action shape: (1000100, 2)
        - Action mean: [-0.01136365 -0.00160795]
        - Action std:  [0.70731187 0.6887115]
    - Dataset: pointmaze-large-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [1.8227280e+01 1.0928875e+01 5.9924867e-05 1.7535901e-07]
        - Observation std:  [11.593134    7.2386537   0.02676597  0.02496127]
        - Action shape: (1005000, 2)
        - Action mean: [0.00297679 -0.02221723]
        - Action std:  [0.70413345 0.70931256]
    - Dataset: pointmaze-giant-navigate-v0
        - Total samples: 1000500, Subtrajectory length: 2001
        - Observation shape: (1000500, 2)
        - Observation mean: [2.51892452e+01  1.73160381e+01 -1.02453356e-04 -9.73671558e-05]
        - Observation std:  [14.725286   11.648896    0.04031942  0.03863621]
        - Action shape: (1000500, 2)
        - Action mean: [-0.00830816  0.00012539]
        - Action std:  [0.7027886 0.6976617]
    - Dataset: pointmaze-giant-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [2.6124264e+01  1.7934097e+01 -4.2366974e-05 -1.6943675e-05]
        - Observation std:  [16.770395   11.946014    0.02510506  0.02465044]
        - Action shape: (1005000, 2)
        - Action mean: [-0.00038941 -0.00806763]
        - Action std:  [0.70801765 0.70562637]
    - Dataset: pointmaze-teleport-navigate-v0
        - Total samples: 1000100, Subtrajectory length: 1001
        - Observation shape: (1000500, 2)
        - Observation mean: [1.9765738e+01  5.7362795e+00 -1.6682481e-05 -5.1102510e-05]
        - Observation std:  [9.971918   6.44103    0.02823308 0.02780383]
        - Action shape: (1000100, 2)
        - Action mean: [-0.05035752 -0.02637289]
        - Action std:  [0.71056426 0.69628423]
    - Dataset: pointmaze-teleport-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [2.0528719e+01  9.8234177e+00  6.3715575e-05 -6.4280961e-05]
        - Observation std:  [11.406025    7.509961    0.02334532  0.02215875]
        - Action shape: (1005000, 2)
        - Action mean: [-0.04648644 -0.06475478]
        - Action std:  [0.6943906  0.71181977]

    '''
    def __init__(
            self, 
            env='pointmaze-large-stitch-v0',
            horizon=100,
            normalizer='LimitsNormalizer',
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
        self.dataset["observations"] = np.concatenate([self.dataset["qpos"], self.dataset["qvel"]], axis=-1)
        if 'antmaze' in env:
            self.dataset['observations'] = self.dataset['observations'][:, :2]
            # only modeling the x,y positions
        if "navigate" in self.dataset_name:
            if "giant" in self.dataset_name:
                sample_length = 2001
            else:
                sample_length = 1001
        elif "stitch" in self.dataset_name:
            sample_length = 201
        else:
            raise ValueError(f"Invalid Dataset {self.dataset}")

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

if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    import matplotlib.pyplot as plt
    import gymnasium 
    os.chdir("../..")
    # cfg = MagicMock()
    #cfg.dataset = "pointmaze-medium-navigate-v0"
    #cfg.dataset =  "pointmaze-medium-stitch-v0"
    #cfg.dataset = "pointmaze-large-navigate-v0"
    #cfg.dataset =  "pointmaze-large-stitch-v0"
    #cfg.dataset = "pointmaze-giant-navigate-v0"
    #cfg.dataset =  "pointmaze-giant-stitch-v0"
    #cfg.dataset = "pointmaze-teleport-navigate-v0"
    # cfg.dataset =  "pointmaze-teleport-stitch-v0"
    #cfg.save_dir = ".ogbench/datasets" # Using default save_dir, "~/.ogbench/data"
    # cfg.env_id = "pointmaze-giant-v0"
    # if "navigate" in cfg.dataset:
    #     cfg.episode_len = 1000
    # elif "stitch" in cfg.dataset:
    #     cfg.episode_len = 100
    # cfg.gamma = 1.0
    # dataset = OGMaze2dOfflineRLDataset(cfg)
    # exit(1)
    dataset = OGMaze2dOfflineRLDataset()

    ## Dataset Visualization
    obs = dataset.dataset["observations"]
    def convert_maze_string_to_grid(maze_string):
       lines = maze_string.split("\\")
       grid = [line[1:-1] for line in lines]
       return grid[1:-1]
    # if "giant" in cfg.dataset:
    #    maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    # maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
    maze_string = "#########\\#OOOOO#O#\\#O#O#OOO#\\#O#O#####\\#OOO#OOO#\\###O###O#\\#OOOOOOO#\\#O###O###\\#OOO#OOO#\\#O#O#O#O#\\#OOOOO#O#\\#########"
    grid = convert_maze_string_to_grid(maze_string)
    import random
    for k in range(100):

        plt.figure()
        steps = 1001
        idx = random.randint(0, len(dataset) - 1)
        obs = dataset.observations[idx]
        plt.scatter(obs[:, 0] / 4 + 1, obs[:, 1] / 4 + 1, c=np.arange(len(obs)), cmap="Reds")
        #plt.scatter(obs[steps * k : steps * (k+1), 0]/4+1, obs[steps * k : steps * (k+1), 1]/4+1, c=np.arange(len(obs[steps * k : steps * (k+1)])), cmap="Reds")
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    plt.gca().add_patch(square)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().set_facecolor("lightgray")
        plt.gca().set_axisbelow(True)
        plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
        plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
        plt.xlim([0.5, len(grid) + 0.5])
        plt.ylim([0.5, len(grid[0]) + 0.5])
        plt.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
        )
        plt.grid(True, color="white", which="minor", linewidth=4)
        plt.gca().spines["top"].set_linewidth(4)
        plt.gca().spines["right"].set_linewidth(4)
        plt.gca().spines["bottom"].set_linewidth(4)
        plt.gca().spines["left"].set_linewidth(4)
        plt.show()
        # save_path = Path(f"./datasets/offline_rl/viz")
        # save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"/root/samples/{k}.png")
        #print("Done.");exit(1)

    breakpoint()
    # o, a, r, n = dataset.__getitem__(10)
    # print(o.shape, a.shape, r.shape, n.shape)
    # plt.figure()
    # plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds")

    # def convert_maze_string_to_grid(maze_string):
    #     lines = maze_string.split("\\")
    #     grid = [line[1:-1] for line in lines]
    #     return grid[1:-1]
    # import os
    
    # #os.environ['MUJOCO_GL'] = 'egl'
    # #if 'SLURM_STEP_GPUS' in os.environ:
    # #    os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
    # #ogbench.locomaze.maze.make_maze_env('point','maze',maze_type='giant')
    # maze_string = "################\\#O#OOOOOO##OOOO#\\#O#O##O#O#OO##O#\\#OOO#OO#OOO#OOO#\\#O###O######O#O#\\#OOO#OOO#OOOO#O#\\###O#O#OO#O#O###\\#OOO#OO#OOO#OOO#\\#O#O#O######O#O#\\#O###OOO#OOO##O#\\#OOOOO#OOO#OOOO#\\################"
    # grid = convert_maze_string_to_grid(maze_string)