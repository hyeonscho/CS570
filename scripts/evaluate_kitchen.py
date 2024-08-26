import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
import diffuser.utils as utils

import os
import torch
import random

import numpy as np
import gym
from ml_logger import logger
import importlib
import pdb


def import_config(config_name):
    module_path = f"detail_configs.{config_name}"
    try:
        module = importlib.import_module(module_path)
        return module.Config
    except ImportError:
        print(f"Error: Module '{config_name}' not found or has no 'Config' attribute.")
        return None


Config = import_config("kitchen_partial")

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
    stitch=True,
)
render_config = utils.Config(
    Config.renderer,
    savepath="render_config.pkl",
    env=Config.dataset,
)

dataset = dataset_config()
renderer = render_config()
loadpath = os.path.join(Config.bucket, Config.dataset, Config.prefix, "checkpoint")
print("\n\nloadpath = ", loadpath, end="\n\n")

loadpath = os.path.join(loadpath, f"state_1000000.pt")
state_dict = torch.load(loadpath, map_location=Config.device)

torch.backends.cudnn.benchmark = True
utils.set_seed(Config.seed)
random.seed(Config.seed)

obs_dim = observation_dim = dataset.observation_dim
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
# logger.print(utils.report_parameters(model), color='green')
trainer.step = state_dict["step"]
trainer.model.load_state_dict(state_dict["model"])
trainer.ema_model.load_state_dict(state_dict["ema"])

num_eval = 10
env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
dones = [False for _ in range(num_eval)]
episode_rewards = [0 for _ in range(num_eval)]

assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w


horizon = Config.horizon
device = Config.device
dynamics_deviate = 0.5
test_ret = Config.test_ret
sample_optim_batch = 1000
dreamer_similarity = 1.5
stitch_L = 30
stitch_R = 60
obs_dim = observation_dim
dream_len = Config.dream_len
stitch_batch_size = 400
returns = to_device(Config.test_ret * torch.ones(num_eval, 1), Config.device)

total_reward = 0
obs_list = [env.reset()[None, :30] for env in env_list]
obs = np.stack(obs_list, axis=0)
while sum(dones) < num_eval:

    pdb.set_trace()
    state_normed = dataset.normalizer.normalize(obs, "observations")

    cond = np.ones(shape=(num_eval, horizon, 2 * obs_dim))
    cond[:, :, obs_dim:] = 0
    cond[:, :1, :obs_dim] = 0
    cond[:, :1, obs_dim:] = state_normed

    conditions = torch.tensor(cond).to(device)

    samples = trainer.ema_model.conditional_sample(
        conditions, returns=returns, verbose=False
    )  # shape is [1, 100, 11]
    obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
    obs_comb = obs_comb.reshape(-1, 2 * obs_dim)
    action = trainer.ema_model.inv_model(obs_comb)

    samples = to_np(samples)
    action = to_np(action)
    pdb.set_trace()
    action = dataset.normalizer.unnormalize(action, "actions")

    obs_list = []
    for i in range(num_eval):
        this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
        obs_list.append(this_obs[None, :30])

        if not dones[i]:
            episode_rewards[i] += this_reward

            if this_done:
                dones[i] = 1
                logger.print(f"Episode ({i}): {episode_rewards[i]}", color="green")

    obs = np.stack(obs_list, axis=0)

episode_rewards = np.array(episode_rewards)

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
