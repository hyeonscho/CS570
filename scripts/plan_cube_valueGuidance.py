import os
import sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import TrueValueGuidedCubePolicy, TrueValueGuidedCube
import diffuser.datasets as datasets
import diffuser.utils as utils

import argparse

class Parser(utils.Parser):
    dataset: str = ''
    config: str = ''

#---------------------------------- setup ----------------------------------#
n_samples = 50
args = Parser().parse_args('plan', add_extras=True)
args.savepath = args.savepath[:-1] + 'value_guidance'
save_path = args.savepath
restricted_pd = args.restricted_pd

from diffuser.utils.serialization import mkdir
mkdir(save_path)
env = datasets.load_environment(args.dataset)

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
guide = TrueValueGuidedCube('every', diffusion.horizon)
policy = TrueValueGuidedCubePolicy(guide, diffusion, dataset.normalizer)

# DQL Load
from dql_cube.main_Cube import hyperparameters
from dql_cube.agents.ql_diffusion import Diffusion_QL as Agent
params = hyperparameters[args.dataset]
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
num_cubes =  (state_dim - 19) // 9

agent = Agent(
    env_name=args.dataset,
    state_dim=state_dim*2,
    action_dim=action_dim,
    max_action=max_action,
    device=0,
    discount=0.99,
    tau=0.005,
    max_q_backup=params["max_q_backup"],
    beta_schedule="vp",
    n_timesteps=5,
    eta=params["eta"],
    lr=params["lr"],
    lr_decay=False,
    lr_maxt=params["num_epochs"],
    grad_norm=params["gn"],
    #goal_dim=params["goal_dim"],
    goal_dim= 3 * num_cubes,
    lcb_coef=4.0,
)

if args.dataset == "cube-single-play-v0":
    dql_folder = "cube-single-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|3|1.0|False|cql_antmaze|0.2|4.0|10"
    agent.load_model(os.path.join(os.getcwd(), "logs", "cube", dql_folder), id=200)
elif args.dataset == "cube-double-play-v0":
    dql_folder = "cube-double-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|6|1.0|False|cql_antmaze|0.2|4.0|10"
    agent.load_model(os.path.join(os.getcwd(), "logs", "cube", dql_folder), id=2000)
elif args.dataset == "cube-triple-play-v0":
    dql_folder = "cube-triple-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|9|1.0|False|cql_antmaze|0.2|4.0|10"
    agent.load_model(os.path.join(os.getcwd(), "logs", "cube", dql_folder), id=2000)
elif args.dataset == "cube-quadruple-play-v0":
    dql_folder = "cube-quadruple-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|12|1.0|False|cql_antmaze|0.2|4.0|10"
    agent.load_model(os.path.join(os.getcwd(), "logs", "cube", dql_folder), id=2000)

max_planning_steps = args.horizon #env.max_episode_steps

scores = []
success_rate = []

for i in range(n_samples):
    observation, info = env.reset(options=dict(task_id = (i % 5) + 1))
    # env.set_task(task_id = (i % 5) + 1)
    # observation, info = env.reset()
    
    goal = info['goal']
    # obs = obs.reshape(1, -1)
    # goal = np.array(goal).reshape(1, -1)
    start = np.concatenate([observation[19+k*9:19+k*9+3] for k in range(num_cubes)], axis=-1)
    goal = np.concatenate([goal[19+k*9:19+k*9+3] for k in range(num_cubes)], axis=-1)
    # target = env.cur_goal_xy # env.xy_to_ij(env.cur_goal_xy)

    # planning에 필요한 cond
    cond = {diffusion.horizon - 1: goal, 0: start}

    rollout = [observation.copy()]
    total_reward = 0
    distance_threshold = 0.3

    _, samples = policy(cond, batch_size=args.batch_size)
    plan = samples.observations
    sequence = plan[0]
    subgoal_pos = 0
    step = 0

    while True:
        if diffusion.horizon - subgoal_pos < 10:
            subgoal = sequence[-1]
        else:
            subgoal = sequence[subgoal_pos]
        action = agent.sample_action(observation, subgoal)
        action = np.clip(action, -1, 1)
        next_observation, reward, terminal, _, _ = env.step(action)
        next_obs = np.concatenate([next_observation[19+k*9:19+k*9+3] for k in range(num_cubes)], axis=-1)
        step += 1
        if np.linalg.norm(next_obs - subgoal) < distance_threshold:
            subgoal_pos += 10
        total_reward += reward
        rollout.append(next_observation.copy())

        if terminal:
            break

        if step > env.max_episode_steps: # truncated
            break
                
        observation = next_observation
    
    rollout_pos = np.array(rollout)
    rollout_pos = np.concatenate([rollout_pos[:, 19+k*9:19+k*9+3] for k in range(num_cubes)], axis=-1)
    plan_rollout = [np.array(plan)[0], np.array(rollout_pos)]
    renderer.composite(join(args.savepath, f'plan_rollout{i}.png'), plan_rollout, ncol=2)

    print(f" {i} / {n_samples}\t t: {step} | r: {reward:.2f} |  R: {total_reward:.2f} | ")

    json_path = join(args.savepath, f"idx{i}_rollout.json")
    json_data = {'step': step, 'return': total_reward, 'term': terminal,
                 'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    success_rate.append(total_reward > 0)

_success_rate = np.sum(success_rate)/n_samples*100
print(f"success rate: {_success_rate:.2f}%")
json_path = join(args.savepath, "success_rate.json")
json.dump({'success_rate:': _success_rate}, open(json_path, 'w'))
