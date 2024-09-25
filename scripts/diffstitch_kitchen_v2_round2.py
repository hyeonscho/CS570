import shutil
import argparse
import time
import pickle
import sys
import os
import torch
import random
import importlib
from collections import defaultdict
import pdb
import numpy as np
import gym
from time import time
from collections import defaultdict
from sklearn.cluster import KMeans
from ml_logger import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.mopo.utils_trans import RewardPredictingModel
import diffuser.utils as utils


def import_config(config_name):
    module_path = f"detail_configs.{config_name}"
    try:
        module = importlib.import_module(module_path)
        return module.Config
    except ImportError:
        print(f"Error: Module '{config_name}' not found or has no 'Config' attribute.")
        return None


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")
        pdb.set_trace()
    # return [x.to(device) for x in xs]


def process_kitchen_episode(episode):
    rewards_ = episode["rew"][1:] - episode["rew"][:-1]
    rewards = np.concatenate([episode["rew"][:1], rewards_], axis=0)
    episode["rew"] = rewards

    obs = episode["obs"]
    episode["obs"] = obs[:, :30]  # remove the goal
    next_obs = episode["next_obs"]
    episode["next_obs"] = next_obs[:, :30]
    return episode


def task_dataset(seq_dataset, task_len):

    tasks = []

    task_data = defaultdict(list)
    rewards = seq_dataset["rew"]

    l = rewards.shape[0]
    for i in range(l):

        for k in seq_dataset:
            task_data[k].append(seq_dataset[k][i])

        if (i + 1) % task_len == 0:
            for k in task_data:
                task_data[k] = np.array(task_data[k])
            tasks.append(task_data)
            task_data = defaultdict(list)

    return tasks


def load_data(data_file):
    with open(data_file, "rb") as fp:
        ds = pickle.load(fp)
    return ds


class LoadSequenceDataset:
    def __init__(self, env_name, data_file, task_len):
        if "kitchen" not in env_name:
            dataset = load_data(data_file)
            if "AntMaze_UMaze-v4" in env_name:
                dataset_name = "antmaze-umaze-v0"
            elif "AntMaze_Medium-v4" in env_name:
                dataset_name = "antmaze-medium-v0"
            elif "AntMaze_Large-v4" in env_name:
                dataset_name = "antmaze-large-v0"
            elif "PointMaze_UMaze-v3" in env_name:
                dataset_name = "pointmaze-umaze-v0"
            elif "PointMaze_Medium-v3" in env_name:
                dataset_name = "pointmaze-medium-v1"
            elif "PointMaze_Large-v3" in env_name:
                dataset_name = "pointmaze-large-v1"
            else:
                raise NotImplementedError
            max_path_length = 600
        else:
            env = gym.make(env_name)
            dataset = env.get_dataset()
            max_path_length = 280

        self.data_size = dataset["rewards"].shape[0]
        print(f"\nnumber of offline data is {dataset['rewards'].shape[0]}\n")

        paths = []
        current_path = {"obs": [], "act": [], "rew": [], "dones": [], "next_obs": []}

        use_timeouts = "timeouts" in dataset

        print(use_timeouts)
        move = []

        for i in range(dataset["rewards"].shape[0] - 1):
            if "kitchen" not in env_name:
                current_path["obs"].append(dataset["observations"]["observation"][i])
                current_path["dones"].append(dataset["terminations"][i])
                current_path["next_obs"].append(
                    dataset["observations"]["observation"][i + 1]
                )
            else:
                current_path["obs"].append(dataset["observations"][i])
                current_path["dones"].append(dataset["terminals"][i])
                current_path["next_obs"].append(dataset["observations"][i + 1])

            current_path["act"].append(dataset["actions"][i])
            current_path["rew"].append(dataset["rewards"][i])

            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = len(current_path["obs"]) == max_path_length

            done = (
                dataset["terminations"][i]
                if "kitchen" not in env_name
                else dataset["terminals"][i]
            )
            if bool(done) or final_timestep:
                for _ in current_path:
                    current_path[_] = np.array(current_path[_])

                if current_path["obs"].shape[0] > 1:
                    move.append(
                        np.linalg.norm(
                            current_path["obs"][1:, :30]
                            - current_path["obs"][:-1, :30],
                            axis=-1,
                        )
                    )
                if "kitchen" in env_name:
                    current_path = process_kitchen_episode(current_path)
                    task_path = task_dataset(current_path, task_len)
                    paths.extend(task_path)
                else:
                    paths.append(current_path)
                current_path = {
                    "obs": [],
                    "act": [],
                    "rew": [],
                    "dones": [],
                    "next_obs": [],
                }
        self.paths = paths
        self.num_traj = len(self.paths)
        self.move_mean = np.concatenate(move, axis=0).mean(0)
        self.move_std = np.concatenate(move, axis=0).std(0)

    def get_full_info_traj(self, idx, gamma=0.99):
        obs = []
        act = []
        rew = []
        next_obs = []
        dones = []
        region_idx = []
        obs = np.copy(self.paths[idx]["obs"])
        act = np.copy(self.paths[idx]["act"])
        rew = np.copy(self.paths[idx]["rew"])
        next_obs = np.copy(self.paths[idx]["next_obs"])
        dones = np.copy(self.paths[idx]["dones"])

        total_return = np.sum(rew)
        discounted_return = 0
        for i in range(rew.shape[0] - 1, -1, -1):
            discounted_return = discounted_return * gamma + rew[i]
        return {
            "obs": obs,
            "act": act,
            "rew": rew,
            "next_obs": next_obs,
            "dones": dones,
            "total_return": total_return,
            "discounted_return": discounted_return,
            "horizon": rew.shape[0],
            "trajectory_idx": idx,
        }


class OptimalBuffer:
    def __init__(self, horizon, ratio=0.1, gamma=0.99):
        self.gamma = gamma
        self.ratio = ratio
        self.info = []
        self.horizon = horizon
        self.region_map = defaultdict(list)
        self.returns_list = []
        self.num_trj = 0

    def insert_traj(self, info):
        current_total_reward = 0
        current_discounted_reward = 0
        for i in range(info["horizon"] - 1, -1, -1):
            current_total_reward += info["rew"][i]
            current_discounted_reward = (
                current_discounted_reward * self.gamma + info["rew"][i]
            )
            if info["horizon"] - i >= self.horizon:
                current_info = {
                    "discounted_reward": current_discounted_reward,
                    "obs": info["obs"][i : i + self.horizon],
                    "segment_idx": i,
                    "traj_idx": info["trajectory_idx"],
                    "rew": info["rew"][i : i + self.horizon],
                }
                self.info.append(current_info)
                self.num_trj += 1

    def finalize(
        self,
    ):
        num_seg = len(self.info)
        all_obs = np.array([info["obs"] for info in self.info]).mean(axis=-1)
        kmeans = KMeans(n_clusters=10, n_init="auto").fit(all_obs)
        labels = kmeans.labels_
        for i in range(num_seg):
            self.info[i]["region_idx"] = str(labels[i])
            self.region_map[str(labels[i])].append(i)
            self.returns_list.append(sum(self.info[i]["rew"]))
        self.returns_list = np.array(self.returns_list)
        self.returns_list -= self.returns_list.max()
        self.returns_list = np.exp(self.returns_list)  # make if positive

    def gen_sample_idx(self, optim_batch, region_avoid, uniform=False):
        sample_idx = []

        for k in self.region_map:
            if k not in region_avoid:
                sample_idx += self.region_map[k]

        if uniform:
            sample_p = None
        else:
            sample_p = self.returns_list[np.array(sample_idx)]
            sample_p = sample_p / sample_p.sum()
        if len(sample_idx) < optim_batch:
            optim_batch = len(sample_idx)

        batch_index = np.random.choice(
            sample_idx, size=optim_batch, replace=False, p=sample_p
        )
        return batch_index

    def repeat_sample_idx(batch_index, repeat):
        return np.repeat(batch_index, repeat)

    def sample_batch_traj(self, optim_batch, region_avoid, uniform=False):
        # stored in (d-rtg, obs:(100, obs_dim))

        batch_index = self.gen_sample_idx(optim_batch, region_avoid, uniform=uniform)

        batch_info = [self.info[_] for _ in batch_index]

        return batch_info


def cosine_similarity(x, y):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    similarity = np.dot(x / x_norm, (y / y_norm).transpose())
    return similarity


def cosine_similarity_pt(x, y):
    x_norm = torch.linalg.norm(x, axis=1, keepdims=True)
    y_norm = torch.linalg.norm(y, axis=1, keepdims=True)
    similarity = torch.matmul(
        x / x_norm,
        (y / y_norm).transpose(0, 1),
    )
    return similarity


def comp_distance(x, y):
    return np.linalg.norm(x - y, axis=-1)


class Timer:
    def __init__(self):
        self.total_clapsed = 0
        self.start = time()
        self.call_time = 0

    def __call__(self):

        clapsed = time() - self.start
        self.call_time += 1

        self.total_clapsed += clapsed
        self.start = time()
        return self.total_clapsed

    def mean(
        self,
    ):

        clapsed = time() - self.start
        self.call_time += 1

        self.total_clapsed += clapsed
        self.start = time()
        return self.total_clapsed / self.call_time


def get_conditions(conditions, start, x, device):

    obs_dim = x.shape[-1]

    conditions[:, :, obs_dim:] = 0
    conditions[:, :start, :obs_dim] = 0
    conditions[:, :start, obs_dim:] = x[:, -start:, :]
    conditions = torch.tensor(conditions).to(device)

    return conditions


def stitch(
    Config, stitch_buffer, data_buffer, dataset, trainer, dynamics_model, env_dataset
):
    horizon = Config.horizon
    device = Config.device
    dynamics_deviate = Config.dynamics_deviate
    test_ret = Config.test_ret
    sample_optim_batch = min(10000, data_buffer.num_trj)
    dreamer_similarity = Config.dreamer_similarity
    stitch_L = 0
    stitch_R = 5
    obs_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    dream_len = Config.dream_len
    stitch_batch_size = min(6000, stitch_buffer.num_trj)

    gen_size = env_dataset.data_size
    move_mean = env_dataset.move_mean
    move_std = env_dataset.move_std
    std_coef = Config.std_coef

    aug_list = []

    timer = Timer()
    timer_2 = Timer()
    generated_transitions = 0
    while generated_transitions < gen_size:
        timer_2()

        trj1_info = stitch_buffer.sample_batch_traj(stitch_batch_size, [], uniform=True)

        trj1_obs = np.stack([info["obs"] for info in trj1_info], axis=0)
        trj1_normed_obs = dataset.normalizer.normalize(trj1_obs, "observations")

        # conditional generation from the end of trj1
        cond = np.ones(shape=(stitch_batch_size, horizon, 2 * obs_dim))
        conditions = get_conditions(cond, dream_len, trj1_normed_obs, device)
        returns = to_device(test_ret * torch.ones(stitch_batch_size, 1), device)

        samples = trainer.ema_model.conditional_sample(
            conditions, returns=returns, verbose=False
        )  # shape is [N, 100, 11]
        normed_gen_obs = to_np(samples)
        gen_obs = dataset.normalizer.unnormalize(normed_gen_obs, "observations")
        gen_obs = torch.tensor(gen_obs).to(device)

        #
        cond_p = np.ones(shape=(stitch_batch_size, horizon, 2 * obs_dim))
        mark_succ = np.zeros(shape=(stitch_batch_size,))
        positions = []
        for i in range(0, stitch_batch_size):
            region_idx1 = trj1_info[i]["region_idx"]

            # sample a batch of optimal trajectories
            available_positions = []
            n_try = 0
            while len(available_positions) == 0:
                candidates_trj2_info = data_buffer.sample_batch_traj(
                    sample_optim_batch, []
                )

                candidates_trj2_obs = np.stack(
                    [info["obs"] for info in candidates_trj2_info], axis=0
                )
                candidates_trj2_obs = torch.tensor(candidates_trj2_obs).to(device)

                # (stitch_L - stitch_R) x sample_optim_batch
                cosin_sim = (
                    cosine_similarity_pt(
                        gen_obs[i, stitch_L:stitch_R],
                        candidates_trj2_obs[:, 0],
                    )
                    .cpu()
                    .numpy()
                )
                # cosin_sim = cosine_similarity(
                #     gen_obs[i, stitch_L:stitch_R], candidates_trj2_obs[:, 0]
                # )
                # see which trajectory has a similarity greater than threshold
                for cur_pos in range(stitch_L, stitch_R):

                    cur_sim = cosin_sim[cur_pos - stitch_L]

                    if np.max(cur_sim) < dreamer_similarity:
                        continue
                    temp_pos = list(np.where(cur_sim >= dreamer_similarity)[0])
                    for pos in temp_pos:
                        available_positions.append(pos)
                n_try += 1
                if n_try > 3:
                    break

            if len(available_positions) == 0:
                # pdb.set_trace()
                positions.append([0, 0])
                continue

            available_positions = list(set(available_positions))

            mark_succ[i] = 1
            chosen_index = random.randint(0, len(available_positions) - 1)
            pos = available_positions[chosen_index]

            trj2_info = candidates_trj2_info[pos]

            traj2_obs = trj2_info["obs"]
            traj2_normed_obs = dataset.normalizer.normalize(traj2_obs, "observations")
            traj2_idx = candidates_trj2_info[pos]["traj_idx"]
            traj2_start_idx = candidates_trj2_info[pos]["segment_idx"]
            region_idx2 = candidates_trj2_info[pos]["region_idx"]

            # (stitch_L - stitch_R) x 1
            sim_i = cosin_sim[:, pos]
            best_pos = np.argmax(sim_i, axis=0) + stitch_L + 1

            start_index = dream_len
            end_index = best_pos - 1
            positions.append(
                [
                    start_index,
                    end_index,
                    traj2_idx,
                    traj2_start_idx,
                    region_idx1,
                    region_idx2,
                ]
            )
            cond_p[i][:, obs_dim:] = 0
            cond_p[i][:start_index, :obs_dim] = 0
            cond_p[i][:start_index, obs_dim:] = trj1_normed_obs[i][-1:]
            cond_p[i][best_pos:, :obs_dim] = 0
            cond_p[i][best_pos:, obs_dim:] = traj2_normed_obs[: horizon - best_pos]

        conditions_p = torch.tensor(cond_p).to(device)
        returns_p = to_device(test_ret * torch.ones(stitch_batch_size, 1), device)
        samples_p = trainer.ema_model.conditional_sample(
            conditions_p, returns=returns_p, verbose=False
        )  # shape is [64, 100, 11]
        np_samples_p = to_np(samples_p)
        obss_p = dataset.normalizer.unnormalize(np_samples_p, "observations")

        traj2_time = timer_2.mean()
        print(f"traj2 preparation time: {traj2_time:.2f}")

        succ_cnt = 0

        print(f" number of candidate trj1: {sum(mark_succ)}")
        if sum(mark_succ) == 0:
            print(f"no trj2 found")
            continue
        # pdb.set_trace()

        total_number_traj = sum(mark_succ)
        filtered_number_traj = 0

        for j in range(0, stitch_batch_size):

            # failed to reach threshold
            if mark_succ[j] != 1:
                print(f"{j}th sample no stitching")
                continue
            end_index = positions[j][1]
            start_index = positions[j][0]
            traj2_idx = positions[j][2]
            traj2_start_idx = positions[j][3]
            region_idx1 = positions[j][4]
            region_idx2 = positions[j][5]

            # move = np.linalg.norm(
            #     obss_p[j][start_index : end_index + 2]
            #     - obss_p[j][start_index - 1 : end_index + 1],
            #     axis=-1,
            # )
            # if (move > move_mean + std_coef * move_std).any() or (
            #     move < move_mean - std_coef * move_std
            # ).any():
            #     print(f"found jumps in {j}th sample")
            #     continue

            best_pos = end_index + 1
            len_path_data = end_index - start_index + 2
            return_info = {}
            return_info["act"] = np.zeros(shape=(len_path_data, action_dim))
            return_info["rew"] = np.zeros(shape=(len_path_data))
            return_info["obs"] = np.zeros(shape=(len_path_data + 1, obs_dim))

            obs_comb = torch.cat(
                [
                    samples_p[j, start_index - 1 : end_index + 1, :],
                    samples_p[j, start_index : end_index + 2, :],
                ],
                dim=-1,
            )
            obs_comb = obs_comb.reshape(-1, 2 * obs_dim)
            action_ = trainer.ema_model.inv_model(obs_comb)  # [1,3]
            normed_actions = to_np(action_)
            actions = dataset.normalizer.unnormalize(normed_actions, "actions")

            dynamics_info = dynamics_model.predict(
                obss_p[j, start_index - 1 : end_index + 1], actions
            )
            pred_obs = dynamics_info["next_obs"]
            pred_rew = dynamics_info["reward"]

            if end_index != 0:
                normed_pred_obs = dataset.normalizer.normalize(pred_obs, "observations")
                pred_deviate = np.linalg.norm(
                    pred_obs - obss_p[j, start_index : end_index + 2], axis=-1
                )
                if (pred_deviate > dynamics_deviate).any():
                    filtered_number_traj += 1
                    deviate = pred_deviate.max()
                    print(f"dynamics: {j}th filtered deviate {deviate:.3f}")
                    continue

            succ_cnt += 1
            return_info["obs"] = obss_p[j][start_index - 1 : end_index + 2]

            aug_cnt = return_info["rew"].shape[0]

            return_info["next_obs"] = return_info["obs"][1:]
            return_info["obs"] = return_info["obs"][:-1]
            return_info["dones"] = np.full((aug_cnt,), False, dtype=bool)
            return_info["region_idx"] = region_idx1 + region_idx2
            return_info["rew"] = pred_rew.squeeze(-1)
            return_info["act"] = actions

            for _ in ["obs", "rew", "dones", "next_obs", "act"]:
                return_info[_] = return_info[_][:]

            trj1_len = trj1_info[j]["segment_idx"] + horizon
            trj2_len = (
                env_dataset.get_full_info_traj(traj2_idx)["obs"].shape[0]
                - traj2_start_idx
                + 1
            )
            generated_transitions = (
                generated_transitions
                + return_info["obs"].shape[0]
                + trj1_len
                + trj2_len
            )
            if succ_cnt % 1 == 0:
                clapsed = timer() / generated_transitions * 1000

                print(
                    f" =========  processed {generated_transitions} transitions,\t clapsed per 1k transition {clapsed:.2f} sec.  ========="
                )

            aug_list.append(
                (
                    trj1_info[j]["traj_idx"],
                    trj1_info[j]["segment_idx"],
                    traj2_idx,
                    traj2_start_idx,
                    start_index,
                    end_index,
                    obss_p[j],
                    return_info["obs"],
                    return_info["act"],
                    return_info["rew"],
                    return_info["next_obs"],
                    return_info["dones"],
                    return_info["region_idx"],
                )
            )

    dataset_name = Config.dataset.replace("-", "_")
    with open(
        f"./kitchen/round2_stitch_{dataset_name}_H{Config.task_len}.pkl", "wb"
    ) as f:
        pickle.dump(aug_list, f)


class StitchDataset:
    def __init__(self, trajectories):
        self.paths = trajectories
        self.num_traj = len(self.paths)
        print(f"processed {self.num_traj} trajectories")

    def get_full_info_traj(self, idx, gamma=0.99):
        obs = []
        act = []
        rew = []
        next_obs = []
        dones = []
        region_idx = []
        obs = np.copy(self.paths[idx]["obs"])
        act = np.copy(self.paths[idx]["act"])
        rew = np.copy(self.paths[idx]["rew"])
        next_obs = np.copy(self.paths[idx]["next_obs"])
        dones = np.copy(self.paths[idx]["dones"])
        region_idx = self.paths[idx]["region_idx"]

        total_return = np.sum(rew)
        discounted_return = 0
        for i in range(rew.shape[0] - 1, -1, -1):
            discounted_return = discounted_return * gamma + rew[i]
        return {
            "obs": obs,
            "act": act,
            "rew": rew,
            "next_obs": next_obs,
            "dones": dones,
            "total_return": total_return,
            "discounted_return": discounted_return,
            "horizon": rew.shape[0],
            "region_idx": region_idx,
            "trajectory_idx": idx,
        }


class StitchBuffer:
    def __init__(self, horizon, ratio=0.1, gamma=0.99):
        self.gamma = gamma
        self.ratio = ratio
        self.info = []
        self.horizon = horizon
        self.region_map = defaultdict(list)
        self.count = 0
        self.returns_list = []
        self.num_trj = 0

    def insert_traj(self, info):
        # only the end half, stitch the end to the start of trj2
        path_len = info["horizon"]
        current_info = {
            "obs": info["obs"][path_len - self.horizon : path_len],
            "segment_idx": path_len - self.horizon,
            "region_idx": info["region_idx"][path_len - self.horizon : path_len],
            "traj_idx": info["trajectory_idx"],
            "rew": info["rew"][path_len - self.horizon : path_len],
        }
        self.info.append(current_info)
        self.region_map[info["region_idx"]].append(self.count)
        self.count += 1
        self.returns_list.append(sum(info["rew"]))
        self.num_trj += 1

    def finalize(
        self,
    ):
        self.returns_list = np.array(self.returns_list)
        self.returns_list -= self.returns_list.max()
        self.returns_list = np.exp(self.returns_list)  # make if positive

    def sample_batch_traj(self, optim_batch, region_avoid, uniform=True):
        # stored in (d-rtg, obs:(100, obs_dim))

        sample_idx = []
        for k in self.region_map:
            if k not in region_avoid:
                sample_idx += self.region_map[k]

        batch_info = []
        batch_index = np.random.choice(sample_idx, size=optim_batch, replace=False)
        batch_info = [self.info[_] for _ in batch_index]

        return batch_info


def process_stitched_trajectories(origin_trj1, stitch_trj, origin_trj2):
    """concatenate one origin_trj and the corresponding stitch_trj"""

    keys = ["obs", "act", "rew", "next_obs", "dones"]
    new_trj = {}
    for k in keys:
        assert k in origin_trj1, f"{k} not in origin_trj1"
        assert k in origin_trj2, f"{k} not in origin_trj2"
        assert k in stitch_trj, f"{k} not in stitch_trj"

        new_trj[k] = np.concatenate(
            [origin_trj1[k], stitch_trj[k], origin_trj2[k]], axis=0
        )
    new_trj["region_idx"] = stitch_trj["region_idx"]
    return new_trj


def main(**cfg):

    Config = import_config(cfg["config"])

    # log path
    BASE_WEIGHTS_PATH = os.path.join(Config.bucket, Config.dataset)
    save_config_path = os.path.join(BASE_WEIGHTS_PATH, "config")
    logger.configure(save_config_path)
    torch.backends.cudnn.benchmark = True

    # training dataset for Diffuser
    dataset_config = utils.Config(
        "datasets.CondSequenceDataset",
        savepath="dataset_config.pkl",
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        data_file=Config.data_file,
        stitch=Config.stitch,
        task_data=Config.task_data,
        aug_data_file=Config.aug_data_file,
        jump=Config.jump,
        segment_return=Config.segment_return,
        jumps=Config.jumps,
        task_len=Config.task_len,
    )

    dataset = dataset_config()
    renderer = None
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    loadpath = os.path.join(Config.bucket, Config.dataset, Config.prefix, "checkpoint")
    print("\n\nloadpath = ", loadpath, end="\n\n")

    # model configs
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    transition_dim = observation_dim

    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
        ll=True,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        n_samples=Config.n_samples,
        train_device=Config.device,
    )
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    # load Diffuser model
    loadpath = os.path.join(loadpath, f"state_800000.pt")
    print(f"load model from {loadpath}")
    state_dict = torch.load(loadpath, map_location=Config.device)
    trainer.step = state_dict["step"]
    trainer.model.load_state_dict(state_dict["model"])
    trainer.ema_model.load_state_dict(state_dict["ema"])

    # load Dynamic model
    dynamics_model = RewardPredictingModel(
        device=Config.device,
        env_name=Config.dataset,
        load_path=Config.dynamic_model_path,
    )

    env_name = Config.dataset
    if "kitchen" in env_name:
        data_file = None
    else:
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

        dataset_name = dataset_name + ".pkl"
        data_file = Config.data_file + dataset_name
    env_dataset = LoadSequenceDataset(
        env_name=Config.dataset, data_file=data_file, task_len=Config.task_len
    )
    data_buffer = OptimalBuffer(horizon=Config.horizon)
    for i in range(env_dataset.num_traj):
        info = env_dataset.get_full_info_traj(i)
        data_buffer.insert_traj(info)
    data_buffer.finalize()
    print(f"data_buffer size: {data_buffer.num_trj}")

    dataset_name = Config.dataset.replace("-", "_")
    with open(
        f"./kitchen/round1_stitch_{dataset_name}_H{Config.task_len}.pkl", "rb"
    ) as f:
        stitched_data = pickle.load(f)
    stitched_trjs = []
    for i in range(len(stitched_data)):

        stitch_trj = dict(
            obs=stitched_data[i][7][1:],
            act=stitched_data[i][8][1:],
            rew=0.0 * stitched_data[i][9][1:],
            next_obs=stitched_data[i][10][1:],
            dones=stitched_data[i][11][1:],
            region_idx=stitched_data[i][12],
        )

        traj2_idx = stitched_data[i][2]
        traj2_segment_idx = stitched_data[i][3]
        trj2 = env_dataset.get_full_info_traj(traj2_idx)
        for k in ["obs", "act", "rew", "next_obs", "dones"]:
            trj2[k] = trj2[k][traj2_segment_idx:]

        traj1_idx = stitched_data[i][0]
        traj1_segment_idx = stitched_data[i][1]
        trj1 = env_dataset.get_full_info_traj(traj1_idx)
        for k in ["obs", "act", "rew", "next_obs", "dones"]:
            trj1[k] = trj1[k][
                : traj1_segment_idx + Config.horizon
            ]  # the last state in trj1 is the first state in stitch_trj

        new_trj = process_stitched_trajectories(trj1, stitch_trj, trj2)
        stitched_trjs.append(new_trj)

    stitch_dataset = StitchDataset(stitched_trjs)
    stitch_buffer = StitchBuffer(horizon=Config.horizon)
    for i in range(stitch_dataset.num_traj):
        info = stitch_dataset.get_full_info_traj(i)
        stitch_buffer.insert_traj(info)
    stitch_buffer.finalize()

    with torch.no_grad():
        stitch(
            Config,
            stitch_buffer,
            data_buffer,
            dataset,
            trainer,
            dynamics_model,
            env_dataset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DiffStitch", add_help=False)
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = vars(parser.parse_args())
    main(**args)
