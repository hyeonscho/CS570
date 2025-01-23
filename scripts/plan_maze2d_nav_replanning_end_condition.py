import os
import sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import argparse

class Parser(utils.Parser):
    dataset: str = ''
    config: str = ''

#---------------------------------- setup ----------------------------------#
n_samples = 50
args = Parser().parse_args('plan', add_extras=True)
args.savepath = args.savepath[:-1] + 'replanning_end_condition'
save_path = args.savepath
restricted_pd = args.restricted_pd

from diffuser.utils.serialization import mkdir
mkdir(save_path)
env = datasets.load_environment(args.dataset)

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
policy = Policy(diffusion, dataset.normalizer)

max_planning_steps = args.horizon #env.max_episode_steps

scores = []
success_rate = []
replanning_at_every = 50

for i in range(n_samples):
    env.set_task(task_id = (i % 5) + 1)
    observation, info = env.reset()
    target = env.cur_goal_xy # env.xy_to_ij(env.cur_goal_xy)

    # planning에 필요한 cond
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    rollout = [observation.copy()]
    total_reward = 0
    distance_threshold = 2
    plan = None
    sequence = None
    k = 0

    for t in range(max_planning_steps):
        if t % replanning_at_every == 0: # start
            if k == 0:
                cond[0] = observation
            else:
                # for j in range(len(rollout)):
                #     cond[j] = rollout[j]
                cond = {
                    0: observation,
                }
                for j in range(len(rollout)-1):
                    cond[diffusion.horizon - (j+1)] = np.array([*target, 0, 0])
            _, samples = policy(cond, batch_size=args.batch_size)
            plan = samples.observations
            sequence = plan[0][:replanning_at_every]
            k = k + 1

        state = env.state_vector().copy()
        if t - (k-1) * replanning_at_every < len(sequence) - 1:
            next_waypoint = sequence[t - (k-1) * replanning_at_every+1]
        else:
            if restricted_pd:
                breakpoint()
                xy = observation.copy()[:2]
                goal = env._target
                target_goal_dist = np.linalg.norm(xy - goal)
                if target_goal_dist <= distance_threshold:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0
                else:
                    next_waypoint = sequence[-2].copy()
            else:
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0

        # Calculate plan_vel and action
        plan_vel = next_waypoint[:2] - state[:2] if t == 0 else next_waypoint[:2] - sequence[t - (k-1) * replanning_at_every -1][:2]
        action = 12.5 * (next_waypoint[:2] - state[:2]) + 1.2 * (plan_vel - state[2:])
        action = np.clip(action, -1, 1)
        next_observation, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        rollout.append(next_observation.copy())

        if terminal:
            break

        observation = next_observation

    plan_rollout = [np.array(plan)[0], np.array(rollout)]
    renderer.composite(join(save_path, f'plan_rollout{i}.png'), plan_rollout, ncol=2)

    print(f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | ")

    json_path = join(save_path, f"idx{i}_rollout.json")
    json_data = {'step': t, 'return': total_reward, 'term': terminal,
                 'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    success_rate.append(total_reward > 0)

_success_rate = np.sum(success_rate)/n_samples*100
print(f"success rate: {_success_rate:.2f}%")
json_path = join(save_path, "success_rate.json")
json.dump({'success_rate:': _success_rate}, open(json_path, 'w'))
