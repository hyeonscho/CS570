import os
import argparse

import torch
import random

import numpy as np
import gym
from ml_logger import logger
import importlib
import pdb
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
import diffuser.utils as utils


def import_config(config_name):
    module_path = f"detail_configs.{config_name}"
    try:
        module = importlib.import_module(module_path)
        return module.Config
    except ImportError:
        print(f"Error: Module '{config_name}' not found or has no 'Config' attribute.")
        return None


def evaluate(Config, test_r, env_list, dataset, trainer):

    horizon = dataset.segmt_len
    device = Config.device
    obs_dim = dataset.observation_dim
    trans_dim = trainer.model.observation_dim
    num_eval = len(env_list)
    num_level = len(dataset.jumps)
    jumps = np.array(dataset.jumps)
    r_ratio = np.append(jumps[:-1] / jumps[1:], 1.0)
    num_eval = len(env_list)

    dones = [False for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    obs_list = [env.reset()[None, :30] for env in env_list]
    obs = np.stack(obs_list, axis=0)

    parital_tasks = ["microwave", "light switch", "slide cabinet", "kettle"]  # parital
    mix_task = ["microwave", "kettle", "bottom burner", "light switch"]
    if "partial" in Config.dataset:
        tasks = parital_tasks
    if "mix" in Config.dataset:
        tasks = mix_task
    comp_task_track = [[] for _ in range(num_eval)]

    while sum(dones) < num_eval:

        returns = to_device(test_r * torch.ones(num_eval, 1), Config.device)
        state_normed = dataset.normalizer.normalize(obs, "observations")
        for l in range(num_level - 1, -1, -1):

            levels = np.eye(num_level, dtype=state_normed.dtype)[l]
            levels = levels.reshape(1, 1, num_level)
            levels = to_device(to_torch(np.tile(levels, (num_eval, horizon, 1))))

            # returns = returns * r_ratio[l]
            cond = np.ones(shape=(num_eval, horizon, 2 * trans_dim))
            cond[:, :, trans_dim:] = 0
            cond[:, :1, :obs_dim] = 0
            cond[:, :1, trans_dim : trans_dim + obs_dim] = state_normed
            if l < num_level - 1:
                level_horizon = int(np.ceil((jumps[l + 1] + 1) / jumps[l]))
                if level_horizon < horizon:
                    cond[:, level_horizon - 1 : level_horizon, :obs_dim] = 0
                    cond[
                        :,
                        level_horizon - 1 : level_horizon,
                        trans_dim : trans_dim + obs_dim,
                    ] = samples_np[:, 1:2, :obs_dim]
                else:
                    cond[:, -1:, :obs_dim] = 0
                    cond[:, -1:, trainer : trans_dim + obs_dim] = samples_np[
                        :, 1:2, :obs_dim
                    ]

            conditions = torch.tensor(cond).to(device)

            samples = trainer.ema_model.conditional_sample(
                conditions, level=levels, returns=returns, verbose=False
            )  # shape is [1, 100, 11]
            samples_np = to_np(samples)

        obs_comb = torch.cat([samples[:, 0, :obs_dim], samples[:, 1, :obs_dim]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2 * obs_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        action = to_np(action)
        action = dataset.normalizer.unnormalize(action, "actions")

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None, :30])

            if not dones[i]:
                episode_rewards[i] += this_reward
                com_task = []
                if this_reward:
                    for task in tasks:
                        if task not in env_list[i].tasks_to_complete:
                            com_task.append(task)
                    comp_task_track[i].append(com_task[:])
                    print(f"env-{i} complete subtasks: {com_task[:]}")

                if this_done:
                    dones[i] = 1
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color="green")

        obs = np.stack(obs_list, axis=0)

    episode_rewards = np.array(episode_rewards)
    print(f" ============== test return:\t {test_r} =================")

    logger.print(
        f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}",
        color="green",
    )
    logger.log_metrics_summary(
        {
            "average_ep_reward": np.mean(episode_rewards),
            "std_ep_reward": np.std(episode_rewards),
        }
    )

    return episode_rewards, comp_task_track


def main(**deps):
    Config = import_config(deps["config"])
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
        act_pad=Config.act_pad,
        cum_rew=Config.cum_rew,
    )

    dataset = dataset_config()
    renderer = None

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    random.seed(Config.seed)

    obs_dim = observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    transition_dim = observation_dim

    num_level = len(Config.jumps)
    if Config.level_dim:
        level_dim = Config.level_dim
    else:
        level_dim = num_level

    if Config.act_pad:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        horizon=dataset.segmt_len,
        transition_dim=transition_dim + level_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
        ll=True,
        level_dim=level_dim,
        level_condition=Config.level_condition,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
        horizon=dataset.segmt_len,
        observation_dim=transition_dim,
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
        level_dim=Config.level_dim,
        num_level=len(Config.jumps),
        level_condition=Config.level_condition,
        act_pad=Config.act_pad,
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
    loadpath = os.path.join(Config.bucket, Config.dataset, Config.prefix, "checkpoint")

    loadpath = os.path.join(loadpath, f"state_1000000.pt")
    print(f"load model from: {loadpath}")

    state_dict = torch.load(loadpath, map_location=Config.device)
    trainer.step = state_dict["step"]
    trainer.model.load_state_dict(state_dict["model"])
    trainer.ema_model.load_state_dict(state_dict["ema"])

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    test_ret = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.10,
        1.2,
    ]
    # test_ret = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    total_rewards = []
    comp_tasks = []
    total_num_eval = 45
    num_eval = 15  # more than 15 instance will raise error
    seeds = [0, 1, 2]
    num_iter = total_num_eval // num_eval
    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    for test_r in test_ret:
        iter_rewards = []
        iter_comp_t = []

        for i in range(num_iter):

            seed = seeds[i]
            utils.set_seed(seed)
            random.seed(seed)
            total_r, comp_t = evaluate(Config, test_r, env_list, dataset, trainer)
            iter_rewards.append(total_r)
            iter_comp_t += comp_t

        iter_rewards = np.concatenate(iter_rewards)
        total_rewards.append(iter_rewards)
        comp_tasks.append(iter_comp_t)
    total_rewards = np.array(total_rewards)
    [env.close() for env in env_list]

    save_path = os.path.join(
        Config.bucket,
        Config.dataset,
        Config.prefix,
        "checkpoint",
        f"evaluation.pkl",
    )

    with open(save_path, "wb") as f:
        pickle.dump(total_rewards, f)

    task_save_path = os.path.join(
        Config.bucket,
        Config.dataset,
        Config.prefix,
        "checkpoint",
        f"task_comp.pkl",
    )
    with open(task_save_path, "wb") as f:
        pickle.dump(comp_tasks, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DiffStitch", add_help=False)
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = vars(parser.parse_args())
    main(**args)
