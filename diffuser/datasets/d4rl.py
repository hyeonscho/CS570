import os
import collections
import numpy as np

# import gym
import h5py
import pdb
import pickle

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# with suppress_output():
#     ## d4rl prints out a variety of warnings
#     import d4rl

# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    if "kitchen" in name:
        import gym
        import d4rl
    else:
        import gymnasium as gym
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_h5_data(aug_path):
    data_dict = {}
    with h5py.File(aug_path, "r", libver="latest", swmr=True) as dataset_file:
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict


def load_data(data_file):
    with open(data_file, "rb") as fp:
        ds = pickle.load(fp)
    return ds


def get_dataset(env, data_path=None):
    if data_path is None:
        dataset = env.get_dataset()
    else:
        if "h5" in data_path:
            dataset = load_h5_data(data_path)
        if "pkl" in data_path:
            dataset = load_data(data_path)

    # if "antmaze" in str(env).lower():
    #     ## the antmaze-v0 environments have a variety of bugs
    #     ## involving trajectory segmentation, so manually reset
    #     ## the terminal and timeout fields
    #     dataset = antmaze_fix_timeouts(dataset)
    #     dataset = antmaze_scale_rewards(dataset)
    #     get_max_delta(dataset)

    return dataset


def sequence_dataset(env, preprocess_fn, data_file=None, task_data=False):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env, data_file)
    dataset = preprocess_fn(dataset)

    N = dataset["rewards"].shape[0]

    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env.max_episode_steps - 1

        for k in ["observations", "rewards", "actions", "terminals"]:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if "maze2d" in env.name:
                episode_data = process_maze2d_episode(episode_data)
            if "kitchen" in env.name and (data_file is None):
                episode_data = process_kitchen_episode(episode_data)
            if task_data:
                tasks = task_dataset(episode_data)
                for task in tasks:
                    yield task
            else:
                yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


# -----------------------------------------------------------------------------#
# -------------------------------- maze2d fixes -------------------------------#
# -----------------------------------------------------------------------------#


def task_dataset(seq_dataset):

    tasks = []

    task_data = collections.defaultdict(list)
    rewards = seq_dataset["rewards"]

    l = rewards.shape[0]
    for i in range(l):

        for k in seq_dataset:
            task_data[k].append(seq_dataset[k][i])

        if rewards[i] == 1:
            for k in task_data:
                task_data[k] = np.array(task_data[k])
            tasks.append(task_data)
            task_data = collections.defaultdict(list)

    if len(task_data) > 0:
        for k in task_data:
            task_data[k] = np.array(task_data[k])
        tasks.append(task_data)
    return tasks


def process_maze2d_episode(episode):
    """
    adds in `next_observations` field to episode
    """
    assert "next_observations" not in episode
    length = len(episode["observations"])
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode


def process_kitchen_episode(episode):
    rewards_ = episode["rewards"][1:] - episode["rewards"][:-1]
    rewards = np.concatenate([episode["rewards"][:1], rewards_], axis=0)
    episode["rewards"] = rewards

    obs = episode["observations"]
    episode["observations"] = obs[:, :30]  # remove the goal
    return episode
