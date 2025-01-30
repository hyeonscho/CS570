import os
import sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import TrueValueGuidedAntmazePolicy_RS, TrueValueGuideAntmaze
import diffuser.datasets as datasets
import diffuser.utils as utils

import argparse

class Parser(utils.Parser):
    dataset: str = ''
    config: str = ''

#---------------------------------- setup ----------------------------------#

n_samples = 50
args = Parser().parse_args('plan', add_extras=True)
args.savepath = args.savepath[:-1] + 'value_guidance_RS_multi_scale_correct_1500'
save_path = args.savepath
restricted_pd = args.restricted_pd

RS_samples = 40 # number of random search samples
if 'giant' in args.dataset:
    RS_samples = 70
else:
    RS_samples = 60
print('Random Search #Samples:', RS_samples)

# RS_samples = 40 # number of random search samples
def sample_and_rank(policy, cond, scale):
    _, samples = policy(cond, batch_size=RS_samples, scale=scale) 

    plans = samples.observations # [RS_samples, horizon, observation_dim]
    values = np.zeros(RS_samples)
    
    for s in range(len(plans)):
        plan = plans[s]
        for t in range(1, plan.shape[0]):
            pos_diff = np.linalg.norm(plan[t, :2] - plan[t-1, :2], axis=-1)
            if pos_diff > 1.0:
                values[s] = 0
                break
            if np.linalg.norm(plan[t, :2] - cond[diffusion.horizon - 1][:2], axis=-1) < 1.0:
                values[s] = (diffusion.horizon - t) / diffusion.horizon
                break

    best_plan_idx = np.argmax(values)

    plan = plans[best_plan_idx][None, ...]

    return plan

from diffuser.utils.serialization import mkdir
mkdir(save_path)
env = datasets.load_environment(args.dataset)

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
guide = TrueValueGuideAntmaze('every', diffusion.horizon)
policy = TrueValueGuidedAntmazePolicy_RS(guide, diffusion, dataset.normalizer)

# DQL Load
from dql.main_Antmaze import hyperparameters
from dql.agents.ql_diffusion import Diffusion_QL as Agent
params = hyperparameters[args.dataset]
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = Agent(
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
    goal_dim=2,
    lcb_coef=4.0,
)

if args.dataset == "antmaze-medium-navigate-v0":
    dql_folder = "antmaze-medium-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
elif args.dataset == "antmaze-large-navigate-v0":
    dql_folder = "antmaze-large-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
elif args.dataset == "antmaze-giant-navigate-v0":
    dql_folder = "antmaze-giant-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
elif args.dataset == "antmaze-teleport-navigate-v0":
    dql_folder = "antmaze-teleport-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
agent.load_model(os.path.join(os.getcwd(), "logs", "dql", dql_folder), id=200)


max_planning_steps = args.horizon #env.max_episode_steps
scales = [0.01, 0.05, 0.1, 0.2, 0.3]
scores = []
success_rate = []

for i in range(n_samples):
    env.set_task(task_id = (i % 5) + 1)
    # scale = scales[(i // 5) % 5] # 순서대로 00000, 11111, 22222, 33333, 44444, 00000, 11111, 22222, 33333, 44444, 00000.
    # print(scale)
    scale = [scales[i%5] for i in range(RS_samples)]
    observation, info = env.reset()
    target = env.cur_goal_xy # env.xy_to_ij(env.cur_goal_xy)

    # planning에 필요한 cond
    cond = {
        diffusion.horizon - 1: np.array([*target]),
        0: np.array([*(observation[:2])])
    }

    rollout = [observation.copy()]
    total_reward = 0

    plan = sample_and_rank(policy, cond, scale)

    # _, samples = policy(cond, batch_size=args.batch_size)
    # plan = samples.observations
    sequence = plan[0]
    subgoal_pos = 0
    step = 0

    while True:
        if diffusion.horizon - subgoal_pos < 10:
            subgoal = sequence[-1] # 마지막 -> DF에서는 10 단위로 해서 문제가 없었던건지 질문하기!
        else:
            subgoal = sequence[subgoal_pos]
        action = agent.sample_action(observation, subgoal)
        action = np.clip(action, -1, 1)
        next_observation, reward, terminal, _, _ = env.step(action)
        step += 1
        if np.linalg.norm(next_observation[:2] - subgoal[:2]) < 0.5:
            subgoal_pos += 10
        total_reward += reward
        rollout.append(next_observation.copy())

        if terminal:
            break

        if step > 1500:
            break

        observation = next_observation

    plan_rollout = [np.array(plan)[0], np.array(rollout)]
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
