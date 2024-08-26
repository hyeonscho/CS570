import argparse
import datetime
import os
import random
import importlib


# import d4rl

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
import pickle
import pdb
from copy import copy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mopo")
    parser.add_argument("--task", type=str, default="kitchen-partial-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def qlearning_dataset(env, dataset, terminate_on_end=False):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """

    N = dataset["terminations"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"]["observation"][i].astype(np.float32)
        new_obs = dataset["observations"]["observation"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        # reward = dataset['rewards'][i].astype(np.float32)
        reward = 0.0
        done_bool = bool(dataset["terminations"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env.spec.max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_).astype(np.float32),
        "terminals": np.array(done_),
    }


# class AntmazeWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         """Constructor for the observation wrapper."""
#         gym.ObservationWrapper.__init__(self, env)

#         self.observation_space["observation"] = gym.spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(self.observation_space["observation"].shape[0] + 2,),
#             dtype=np.float64,
#         )
#         self.env = env

#     def reset(
#         self,
#         *,
#         seed=None,
#         options=None,
#     ):
#         obs, info = self.env.reset(seed=seed, options=options)
#         return self.observation(obs), info

#     def step(self, action):
#         """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         return self.observation(observation), reward, terminated, truncated, info

#     def observation(self, observation):
#         """Returns a modified observation.

#         Args:
#             observation: The :attr:`env` observation

#         Returns:
#             The modified observation
#         """
#         observation["observation"] = np.concatenate(
#             (observation["achieved_goal"], observation["observation"]), axis=0
#         )
#         return observation

#     def set_state(self, qpos, qvel):
#         self.env.unwrapped.ant_env.set_state(qpos, qvel)


def train(args=get_args()):
    # create env and dataset
    if "Maze" not in args.task:
        import gym
        import d4rl

        dataset_name = None

    else:
        import gymnasium as gym

        if "AntMaze_UMaze-v4" in args.task:
            dataset_name = "antmaze-umaze-v0"
        elif "AntMaze_Medium-v4" in args.task:
            dataset_name = "antmaze-medium-v0"
        elif "AntMaze_Large-v4" in args.task:
            dataset_name = "antmaze-large-v0"
        elif "PointMaze_UMaze-v3" in args.task:
            dataset_name = "pointmaze-umaze-v1"
        elif "PointMaze_Medium-v3" in args.task:
            dataset_name = "pointmaze-medium-v1"
        elif "PointMaze_Large-v3" in args.task:
            dataset_name = "pointmaze-large-v1"

    env = gym.make(args.task)
    if "AntMaze" in args.task:
        env = AntmazeWrapper(env)

    if "Maze" not in args.task:
        dataset = d4rl.qlearning_dataset(env)
        if "kitchen" in args.task:
            dataset["observations"] = dataset["observations"][:, :30]
            dataset["next_observations"] = dataset["next_observations"][:, :30]
            obs_space = copy(env.observation_space)
            obs_space.shape = (30,)
            args.obs_space = obs_space
            args.obs_shape = obs_space.shape
        else:
            args.obs_space = env.observation_space
            args.obs_shape = args.obs_space.shape

    else:
        data_file = (
            f"/common/users/cc1547/dataset/rainbow/stitching_maze/{dataset_name}.pkl"
        )
        with open(data_file, "rb") as fp:
            ds = pickle.load(fp)
        dataset = qlearning_dataset(env, ds)
        args.obs_shape = env.observation_space["observation"].shape
        args.obs_space = env.observation_space["observation"]

    args.action_dim = np.prod(env.action_space.shape)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # env.seed(args.seed)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    args.logdir = "/common/users/cc1547/projects/rainbow/diffstitch/dynamic"
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer, log_path=log_path)

    Devid = 0 if args.device == "cuda" else -1
    set_device_and_logger(Devid, logger)

    # import configs
    if "AntMaze" in args.task or "PointMaze" in args.task:
        task = "maze"
        import_path = f"static_fns.{task}"
    elif "kitchen" in args.task:
        task = "kitchen"
        import_path = f"static_fns.{task}"
    else:
        task = args.task.split("-")[0]
        import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    # create policy model
    # actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    # critic1_backbone = MLP(
    #     input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256]
    # )
    # critic2_backbone = MLP(
    #     input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256]
    # )
    # dist = DiagGaussian(
    #     latent_dim=getattr(actor_backbone, "output_dim"),
    #     output_dim=args.action_dim,
    #     unbounded=True,
    #     conditioned_sigma=True,
    # )

    # actor = ActorProb(actor_backbone, dist, args.device)
    # critic1 = Critic(critic1_backbone, args.device)
    # critic2 = Critic(critic2_backbone, args.device)
    # actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    # critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = (
            args.target_entropy
            if args.target_entropy
            else -np.prod(env.action_space.shape)
        )

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    # sac_policy = SACPolicy(
    #     actor,
    #     critic1,
    #     critic2,
    #     actor_optim,
    #     critic1_optim,
    #     critic2_optim,
    #     action_space=env.action_space,
    #     dist=dist,
    #     tau=args.tau,
    #     gamma=args.gamma,
    #     alpha=args.alpha,
    #     device=args.device,
    # )

    # create dynamics model
    dynamics_model = TransitionModel(
        obs_space=args.obs_space,
        action_space=env.action_space,
        static_fns=static_fns,
        lr=args.dynamics_lr,
        **config["transition_params"],
    )

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size
        * args.rollout_length
        * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
    )

    # create MOPO algo
    algo = MOPO(
        None,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"],
    )

    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        eval_episodes=args.eval_episodes,
    )

    # pretrain dynamics model on the whole dataset
    trainer.train_dynamics()


if __name__ == "__main__":
    train()
