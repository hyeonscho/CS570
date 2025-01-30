import torch
import numpy as np
import os
from collections import namedtuple
from .normalization import DatasetNormalizer
from omegaconf import DictConfig
Batch = namedtuple("Batch", "trajectories conditions")

class VOGMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    '''
    - Dataset: visual-pointmaze-medium-navigate-v0
        - Observation mean: [0.53332597, -0.57663816, -0.15480594, -0.10989726,  0.13822828, -0.7565398 , -0.67368555, -0.5261524]
        - OBservation std: [2.230295 , 1.8695153, 2.5765393, 2.5024776, 2.409886 , 2.3264396, 2.2680814, 2.1177504]
        - Position mean: [10.273524  9.648321]
        - Position std: [5.627576 4.897987]
        - Action mean: [-0.00524961 -0.00168911]
        - Action std:  [0.70124096 0.6971626]
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
            only_start_condition = False
        ): #self,cfg: DictConfig , split: str = "training"):

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
        # self.env_id = cfg.env_id
        self.dataset_name = env
        self.n_frames = self.horizon
        self.gamma = 1.0
        self.split = split
        # self.dataset_url = cfg.dataset_url
        if "giant" in self.dataset_name:
            sample_length = 2001
        else:
            sample_length = 1001
        
        assert self.n_frames <= sample_length, f"Episode length {self.n_frames} is greater than sample length {sample_length}"
        
        # self.dataset_url = cfg.dataset_url
        raw_observations, raw_actions = self.get_dataset()

        self.observation_dim = raw_observations.shape[-1]
        self.action_dim = raw_actions.shape[-1]

        raw_observations = raw_observations.reshape(-1, sample_length, raw_observations.shape[-1])
        raw_actions = raw_actions.reshape(-1, sample_length, raw_actions.shape[-1])
        raw_terminals = np.zeros((raw_observations.shape[0], sample_length))
        raw_terminals[:, -1] = True
        raw_rewards = np.copy(raw_terminals)
        raw_values = self.compute_value(raw_rewards) * (1 - self.gamma) * 4 - 1
        self.observations, self.actions, self.terminals, self.rewards, self.values = [], [], [], [], []
        for i in range(raw_observations.shape[0]):
            for j in range(sample_length - self.n_frames):
                self.observations.append(raw_observations[i, j:j+self.n_frames])
                self.actions.append(raw_actions[i, j:j+self.n_frames])
                self.terminals.append(raw_terminals[i, j:j+self.n_frames])
                self.rewards.append(raw_rewards[i, j:j+self.n_frames])
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.terminals = np.array(self.terminals)
        self.rewards = np.array(self.rewards)

        self.normalizer = DatasetNormalizer({'observations':self.observations.reshape(-1, self.observations.shape[-1]), 'actions':self.actions.reshape(-1, self.actions.shape[-1])}, normalizer)
        self.normalized_obs = self.normalizer.normalize(self.observations.reshape(-1, self.observations.shape[-1]), 'observations').reshape(-1, self.n_frames, self.observations.shape[-1])
        self.normalized_act = self.normalizer.normalize(self.actions.reshape(-1, self.actions.shape[-1]), 'actions').reshape(-1, self.n_frames, self.actions.shape[-1])

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
        # observation = torch.from_numpy(self.normalized_obs[idx]).float() # (episode_len, obs_dim)
        # action = torch.from_numpy(self.normalized_act[idx]).float() # (episode_len, act_dim)
        # conditions = self.get_conditions(self.normalized_obs, self.only_start_condition)
        # if self.jump_action == "none":
        #     trajectories = observation
        # else:
        #     trajectories = np.concatenate([action, observation], axis=-1)
        # batch = Batch(trajectories, conditions)
        return batch
    
    def __len__(self):
        return len(self.observations)

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
    
    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(reward.shape[1] - 2, -1, -1):
            value[:, i] += self.gamma * value[:, i + 1]
        return value
    
    def get_dataset(self):
        # path = os.path.join(self.dataset_url, self.split)
        if 'medium' in self.env:
            path = '/home/baek1127/ogbench/embedded_data/medium/training'
        elif 'large' in self.env:
            path = '/home/baek1127/ogbench/embedded_data/large/training'
        else:
            path =  '/home/baek1127/ogbench/embedded_data/giant/training'
        emb = np.load(os.path.join(path, 'latent.npy'))
        act = np.load(os.path.join(path, 'actions.npy'))
        return emb, act
    
# import torch
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from pathlib import Path
# from omegaconf import DictConfig
# from ogbench.pretrain.models.mlp import MLP

# # (나머지 Dataset 관련 코드는 그대로 두고, visualization 부분만 교체 예시)
# def convert_maze_string_to_grid(maze_string):
#     lines = maze_string.split("\\")
#     grid = [line[1:-1] for line in lines]
#     return grid[1:-1]
# def unnormalize_pos(pos):
#     # 예시용 unnormalize 함수 (사용자 정의)
#     # pos_mean = np.array([10.273524, 9.648321])
#     # pos_std = np.array([5.627576, 4.897987])
#     pos_mean = np.array([16.702621, 10.974173])
#     pos_std = np.array([10.050303, 6.8203936])
#     pos = pos * pos_std + pos_mean
#     pos = pos / 4 + 1
#     print(pos.shape)
#     print(pos)
#     pos = pos[:, [1, 0]]
#     # pos = np.column_stack((7 - pos[:, 1], 7 - pos[:, 0]))
#     print(pos)
#     return pos


# if __name__ == "__main__":
#     from unittest.mock import MagicMock
#     cfg = MagicMock()
#     cfg.dataset = "visual-pointmaze-medium-navigate-v0"
#     # cfg.dataset = "visual-pointmaze-large-navigate-v0"
#     cfg.env_id = "visual-pointmaze-medium-v0"
#     # cfg.env_id = "visual-pointmaze-large-v0"
#     cfg.episode_len = 500
#     cfg.gamma = 1.0
#     # cfg.dataset_url = "/home/hyeons/workspace/HierarchicalDiffusionForcing/data/embedded_data/large"
#     cfg.dataset_url = "/home/baek1127/ogbench/embedded_data/medium"
#     # 1) 데이터셋 불러오기
#     dataset = VOGMaze2dOfflineRLDataset(cfg, split="training")
#     # 2) 미로를 나타내는 문자열과 이를 시각화할 그리드 생성
#     maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
#     # maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
#     grid = convert_maze_string_to_grid(maze_string)
#     # 3) MLP 모델 로드 (embedding -> position)
#     model = MLP(8, 2, 1024, 4)
#     model.load_state_dict(torch.load("/home/baek1127/ogbench/embeded_data/medium/e2s.pth"))
#     model.eval()
#     # 5) 시각화 시작
#     plt.figure(figsize=(6, 6))
#     # (A) 미로의 벽을 검은색 사각형으로 표현
#     for i, row in enumerate(grid):
#         for j, cell in enumerate(row):
#             if cell == "#":
#                 square = plt.Rectangle(
#                     (i + 0.5, j + 0.5),
#                     1, 1,
#                     edgecolor="black",
#                     facecolor="black"
#                 )
#                 plt.gca().add_patch(square)
#     for idx in range(0, 500000, 500):
#         # 예시로 데이터셋에서 하나의 trajectory만 시각화
#         obs, act, rew, nonterm = dataset.__getitem__(idx)
#         # Ensure all arrays are of the same type (float)
#         obs_mean = np.array([0.56942457, -0.7748614, 0.03422518, -0.05451093, 0.00234696, -0.4766496, -0.53628725, -1.1162051], dtype=np.float32)
#         obs_std = np.array([2.0956497, 2.2737527, 2.3882532, 2.6977062, 2.1805387, 2.6994274, 2.4300833, 2.137858], dtype=np.float32)
#         obs = (obs - obs_mean) / obs_std
#         # 4) 모델로부터 위치 디코딩 & 좌표 복원
#         with torch.no_grad():
#             decoded_pos = model(obs)  # shape: (episode_len, 2)
#         decoded_pos = decoded_pos.detach().cpu().numpy()
#         decoded_pos = unnormalize_pos(decoded_pos)
#         # (B) 디코딩된 위치(trajectory)를 빨간색 gradation으로 표시
#         plt.scatter(
#             decoded_pos[:, 0],         # x 좌표
#             decoded_pos[:, 1],         # y 좌표
#             color="red",               # 점 색깔을 빨간색으로 고정
#             s=3,                       # 점 크기 조절
#             alpha=0.005                  # 점의 투명도 조절
#         )
#     # (C) 그래프 스타일 설정: 축 비율, 배경색, 그리드 등
#     plt.gca().set_aspect("equal", adjustable="box")
#     plt.gca().set_facecolor("lightgray")
#     plt.gca().set_axisbelow(True)
#     # 필요하다면, 미로 크기에 따라 그리드/눈금 등 세부 설정
#     # 현재 예제 미로는 8x8
#     plt.xlim([0.5, len(grid) + 0.5])
#     plt.ylim([0.5, len(grid[0]) + 0.5])
#     plt.tick_params(
#         axis="both", which="both",
#         bottom=False, top=False, left=False, right=False,
#         labelbottom=False, labelleft=False
#     )
#     plt.grid(True, color="white", which="major", linewidth=1)
#     # (D) 최종 그림 표시
#     plt.show()
#     # save_path = Path("./my_viz_folder")
#     # save_path.mkdir(parents=True, exist_ok=True)
#     plt.savefig("/root/decoded_trajectory.png")