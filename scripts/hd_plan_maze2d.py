import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import os

import argparse

parser = argparse.ArgumentParser("hd", add_help=False)
parser.add_argument("--cfg", type=str)
parser.add_argument('--rpd', action='store_true')
args, leftovers = parser.parse_known_args()

hl_cfg_wo_config_word = args.cfg.split("config.")[1]
hl_cfg = args.cfg
ll_cfg = args.cfg.replace("hl", "ll")

# I do not know why the parser does not work -> quick solution -> argparse and override
restricted_pd = args.rpd

class HLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    # config: str = "config.maze2d_hl_hmdConfig300_J15"
    config: str = hl_cfg
    restricted_pd: bool = restricted_pd


hl_args = HLParser().parse_args("plan", add_extras=False) # Discovered it later, so had to disable it as I am doing it here, until I test it


class LLParser(utils.Parser):
    dataset: str = "maze2d-large-v1"
    # config: str = "config.maze2d_ll_hmdConfig300_J15"
    config: str = ll_cfg
    restricted_pd: bool = restricted_pd

ll_args = LLParser().parse_args("plan", add_extras=False) # Discovered it later, so had to disable it as I am doing it here, until I test it

# ---------------------------------- setup ----------------------------------#

# ---------------------------------- loading ----------------------------------#


# bad workaround, just for now
hl_args.restricted_pd =  restricted_pd
hl_args.savepath = hl_args.savepath.replace('rpdFalse', f'rpd{restricted_pd}')
hl_args.savepath = hl_args.savepath.replace('rpdTrue', f'rpd{restricted_pd}')


from diffuser.utils.serialization import mkdir
print(hl_args.savepath)
mkdir(hl_args.savepath)


n_samples = 100

loadpath = (hl_args.logbase, hl_args.dataset, hl_args.diffusion_loadpath)

# print(loadpath)
# print(ll_args.logbase, ll_args.dataset, ll_args.diffusion_loadpath)
# exit()
hl_diffusion_experiment = utils.load_diffusion(
    hl_args.logbase,
    hl_args.dataset,
    hl_args.diffusion_loadpath,
    epoch=hl_args.diffusion_epoch,
)
hl_diffusion = hl_diffusion_experiment.ema
dataset = hl_diffusion_experiment.dataset
hl_policy = Policy(hl_diffusion, dataset.normalizer)


ll_diffusion_experiment = utils.load_diffusion(
    ll_args.logbase,
    ll_args.dataset,
    ll_args.diffusion_loadpath,
    epoch=ll_args.diffusion_epoch,
)
ll_diffusion = ll_diffusion_experiment.ema
ll_policy = Policy(ll_diffusion, dataset.normalizer)

env_eval = datasets.load_environment(hl_args.dataset)

total_rewards = []
scores = []
rollouts = []
plans = []
track_action = []


print(f"{hl_args.diffusion_epoch}")
print(hl_args.savepath)

scores = []
for i in range(n_samples):
    observation = env_eval.reset()
    
    target = env_eval._target
    hl_cond = {
        hl_diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    init_obs = observation.copy()
    observation = env_eval._get_obs()
    rollout = [observation.copy()]

    hl_cond[0] = observation
    action, samples = hl_policy(hl_cond, batch_size=hl_args.batch_size)
    hl_plan = samples.observations

    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0],
        ll_args.horizon - 1: ll_cond_[:, -1],
    }

    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    ll_samples = ll_samples.observations
    ll_samples = ll_samples.reshape(B, (M - 1), ll_args.horizon, -1)
    ll_samples = np.concatenate(
        [
            ll_samples[:, 0, :1],
            ll_samples[:, :, 1:].reshape(B, (M - 1) * hl_args.jump, -1),
        ],
        axis=1,
    )
    ll_sequence = ll_samples[0]
    total_reward = []
    action_list = []

    max_episode_steps = env_eval.max_episode_steps
    finished = False
    t = 0
    distance_threshold = 2
    # print(len(ll_sequence))
    while t < max_episode_steps:
        if finished:
            break
        else:
            if t < len(ll_sequence) - 1:
                next_waypoint = ll_sequence[t]
            else:
                if restricted_pd:
                    xy = observation.copy()[:2]
                    goal = env_eval._target
                    target_goal_dist = np.linalg.norm(xy - goal)
                    # print(target_goal_dist)
                    if target_goal_dist <= distance_threshold:
                        next_waypoint = ll_sequence[-1].copy()
                        next_waypoint[2:] = 0
                    else:
                        # print(target_goal_dist, t)
                        next_waypoint = ll_sequence[-2].copy()
                else:
                    # xy = observation.copy()[:2]
                    # goal = env_eval._target
                    # target_goal_dist = np.linalg.norm(xy - goal)
                    # print(target_goal_dist)

                    next_waypoint = ll_sequence[-1].copy()
                    next_waypoint[2:] = 0

            state = observation.copy()
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = env_eval.step(action)
            t += 1
            total_reward.append(reward)
            score = env_eval.get_normalized_score(sum(total_reward))

            ## update rollout observations
            rollout.append(next_observation.copy())
            if terminal or t >= max_episode_steps:
                finished = True
                print(
                    f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {sum(total_reward):.2f} | score: {score:.4f} | "
                )
                break
            observation = next_observation

    rollouts.append(rollout)
    total_rewards.append(total_reward)
    scores.append(env_eval.get_normalized_score(sum(total_reward)))

    ## save result as a json file
    json_path = join(hl_args.savepath, f"idx{i}_rollout.json")
    json_data = {
        "score": score,
        "step": t,
        "return": total_reward,
        "term": terminal,
    }
    json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)
    scores.append(score*100)

print(f"{np.mean(scores):.1f} +/- {np.std(scores)/np.sqrt(len(scores)):.2f}")