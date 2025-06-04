import os
import sys
import time
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import time

from diffuser.guides.policies import TrueValueGuidedPolicy, TrueValueGuide, Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import argparse

class Parser(utils.Parser):
    dataset: str = ''
    config: str = ''

#---------------------------------- setup ----------------------------------#
n_samples = 50
args = Parser().parse_args('plan', add_extras=True)
args.savepath = args.savepath[:-1] + 'time_log'
save_path = args.savepath
restricted_pd = True

from diffuser.utils.serialization import mkdir
mkdir(save_path)

if 'stitched' in args.dataset:
    env = datasets.load_environment(args.dataset.split('stitched-')[1])
else:
    # print(args.dataset) # pointmaze-medium-navigate-v0
    env = datasets.load_environment(args.dataset)

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
diffusion.n_timesteps = args.n_diffusion_steps
guide = TrueValueGuide('every', diffusion.horizon)
policy = TrueValueGuidedPolicy(guide, diffusion, dataset.normalizer)
# policy = Policy(diffusion, dataset.normalizer)

# 여기서 horizon이 100이면 총 1000 step을 위해 10번 더 길게 반복
# max_planning_steps = args.horizon #env.max_episode_steps
max_planning_steps = 500


scores = []
success_rate = []
for i in range(n_samples):
    env.set_task(task_id = (i % 5) + 1)
    observation, info = env.reset()
    target = env.cur_goal_xy # env.xy_to_ij(env.cur_goal_xy)

    # planning에 필요한 cond
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }
    # cond = {}

    rollout = [observation.copy()]
    total_reward = 0
    distance_threshold = 2
    plan = None
    sequence = None
    start_time = time.time()
    for t in range(max_planning_steps):
        if t % diffusion.horizon == 0: # start
            cond[0] = observation

            # # cond[0][:2] = (cond[0][:2]-1)*4
            # cond[diffusion.horizon - 1][:2] = (cond[diffusion.horizon - 1][:2]-1)*4
            # breakpoint()
            _, samples = policy(cond, batch_size=args.batch_size)
            planning_time = time.time() - start_time
            plan = samples.observations
            sequence = plan[0]

        state = env.state_vector().copy()
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            if restricted_pd:
                xy = observation.copy()[:2]
                goal = target
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
        plan_vel = next_waypoint[:2] - state[:2] if t == 0 else next_waypoint[:2] - sequence[t-1][:2]
        action = 12.5 * (next_waypoint[:2] - state[:2]) + 1.2 * (plan_vel - state[2:])
        action = np.clip(action, -1, 1)
        next_observation, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        rollout.append(next_observation.copy())

        if terminal:
            break

        observation = next_observation

    plan_rollout = [np.array(plan)[0], np.array(rollout)]
    renderer.composite(join(args.savepath, f'plan_rollout{i}.png'), plan_rollout, ncol=2)
    print(f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | Time: {planning_time:.2f} | ")

    json_path = join(args.savepath, f"idx{i}_rollout.json")
    json_data = {'step': t, 'return': total_reward, 'term': terminal,
                 'epoch_diffusion': diffusion_experiment.epoch, 'planning_time': planning_time}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    success_rate.append(total_reward > 0)

_success_rate = np.sum(success_rate)/n_samples*100
print(f"success rate: {_success_rate:.2f}%")
json_path = join(args.savepath, "success_rate.json")
json.dump({'success_rate:': _success_rate}, open(json_path, 'w'))

