import os
import sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy, TrueValueGuide, TrueValueGuidedPolicy
import diffuser.datasets as datasets
import diffuser.utils as utils
# pd_restriction_cfg = {'maze2d-large-v1': {'nearer':1, 'near': 2, 'far': 4, 'stay': 0},
#                       'maze2d-giant-v1': {'nearer':1.5, 'near': 3, 'far': 6, 'stay': 0},
#                       'maze2d-xxlarge-v1': {'nearer':2, 'near': 4, 'far': 8, 'stay': 0},
#                       'maze2d-xxlarge-v2': {'nearer':2, 'near': 4, 'far': 8, 'stay': 0},}
# plan_cfg = {'n_samples': 200}
# ---------------------------------- setup ----------------------------------#
class HLParser(utils.Parser):
    dataset: str = ''#'maze2d-large-v1'
    config: str = ''#'config.maze2d_hmdConfig300'


hl_args = HLParser().parse_args("plan", add_extras=True) # Discovered it later, so had to disable it as I am doing it here, until I test it

print("================hl====================")
print("====================================")
print("====================================")
print(hl_args.dataset, hl_args.config)
print(hl_args.savepath)
print("====================================")
print("====================================")
print("====================================")

# restricted_pd = hl_args.restricted_pd

# restricted_pd = False if restricted_pd == 'False' else restricted_pd
# assert restricted_pd in ['near', 'far', 'stay', 'nearer', False]
# pd_restriction_cfg_idx = restricted_pd if restricted_pd else 'near'
# distance_threshold = pd_restriction_cfg[hl_args.dataset][pd_restriction_cfg_idx]
restricted_pd = True
conditional = hl_args.conditional
# print(f"Restricted PD: {restricted_pd}, Conditional: {conditional}, Distance Threshold: {distance_threshold}")


hl_cfg_wo_config_word = hl_args.config.split("config.")[1]
ll_cfg = hl_args.config.replace('high', 'low')
class LLParser(utils.Parser):
    dataset: str = ''
    config: str = ll_cfg

# ll_args = LLParser().parse_args("plan", add_extras=False, specific_config=ll_cfg) # Discovered it later, so had to disable it as I am doing it here, until I test it
ll_args = LLParser().parse_args("plan", add_extras=True, specific_config=ll_cfg)

print("=================ll===================")
print("====================================")
print("====================================")
print(ll_args.dataset, ll_args.config)
print(ll_args.savepath)
print("====================================")
print("====================================")
print("====================================")



# ---------------------------------- loading ----------------------------------#
n_samples = 50

# loadpath = (hl_args.logbase, hl_args.dataset, hl_args.diffusion_loadpath)
# print(loadpath)
# print(ll_args.logbase, ll_args.dataset, ll_args.diffusion_loadpath)
# exit()
print(hl_args.dataset)

env = datasets.load_environment(hl_args.dataset)

hl_diffusion_experiment = utils.load_diffusion(
    hl_args.logbase,
    hl_args.dataset,
    hl_args.diffusion_loadpath,
    epoch=hl_args.diffusion_epoch,
)
hl_diffusion = hl_diffusion_experiment.ema
dataset = hl_diffusion_experiment.dataset
renderer = hl_diffusion_experiment.renderer
# hl_policy = Policy(hl_diffusion, dataset.normalizer)
hl_guide = TrueValueGuide('every', hl_diffusion.horizon)
# hl_policy = TrueValueGuidedPolicy(hl_guide, hl_diffusion, dataset.normalizer)
hl_policy = Policy(hl_diffusion, dataset.normalizer)
print(ll_args.logbase, ll_args.dataset, ll_args.diffusion_loadpath)


ll_diffusion_experiment = utils.load_diffusion(
    ll_args.logbase,
    ll_args.dataset,
    ll_args.diffusion_loadpath,
    epoch=ll_args.diffusion_epoch,
)
ll_diffusion = ll_diffusion_experiment.ema
ll_policy = Policy(ll_diffusion, dataset.normalizer)
ll_guide = TrueValueGuide('every', ll_diffusion.horizon)
# ll_policy = TrueValueGuidedPolicy(ll_guide, ll_diffusion, dataset.normalizer)

# env_eval = datasets.load_environment(hl_args.dataset)
# env = datasets.load_environment(hl_args.dataset)

total_rewards = []
scores = []
rollouts = []
plans = []
track_action = []

print(f"{hl_args.diffusion_epoch}")
print(hl_args.savepath)

scores = []
success_rate = []
for i in range(n_samples):
    # observation = env_eval.reset()
    env.set_task(task_id=(i % 5) + 1)
    observation, info = env.reset()
    target = env.cur_goal_xy  # env.xy_to_ij(env.cur_goal_xy)
    
    # if conditional:
    #     print('Resetting target')
    #     env_eval.set_target()

    
    # target = env_eval._target
    hl_cond = {
        hl_diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    # init_obs = observation.copy()
    # observation = env_eval._get_obs()
    rollout = [observation.copy()]
    total_reward = 0
    distance_threshold = 2
    plan = None
    sequence = None        
    hl_cond[0] = observation
    
    start_time = time.time()
    _, samples = hl_policy(hl_cond, batch_size=hl_args.batch_size)
    hl_time = time.time() - start_time
    hl_plan = samples.observations
    
    B, M = hl_plan.shape[:2]
    ll_cond_ = np.stack([hl_plan[:, :-1], hl_plan[:, 1:]], axis=2)
    ll_cond_ = ll_cond_.reshape(B * (M - 1), 2, -1)
    ll_cond = {
        0: ll_cond_[:, 0], # (B, 4)
        ll_args.horizon - 1: ll_cond_[:, -1], # (B, 4)
    }
    start_time = time.time()
    _, ll_samples = ll_policy(ll_cond, batch_size=-1)
    planning_time = time.time() - start_time + hl_time
    ll_samples = ll_samples.observations
    ll_samples = ll_samples.reshape(B, (M - 1), ll_args.horizon, -1)
    ll_samples = np.concatenate(
        [
            ll_samples[:, 0, :1],
            ll_samples[:, :, 1:].reshape(B, (M - 1) * hl_args.jump, -1),
        ],
        axis=1,
    )
    plan = ll_samples
    ll_sequence = ll_samples[0]
    # total_reward = []
    action_list = []

    max_episode_steps = 500
    finished = False
    t = 0
    first_time_active = 500
    # print(len(ll_sequence))
    for t in range(max_episode_steps):
        # state = env.state_vector().copy()
        if t < len(ll_sequence) - 1:
            next_waypoint = ll_sequence[t]
        else:
            if restricted_pd:
                xy = observation.copy()[:2]
                goal = target
                target_goal_dist = np.linalg.norm(xy - goal)
                # if restricted_pd == 'stay':
                #     next_waypoint = observation.copy()
                #     next_waypoint[2:] = 0
                # else:
                    # print(target_goal_dist)
                if target_goal_dist <= distance_threshold:
                    first_time_active = min(first_time_active, t)
                    next_waypoint = ll_sequence[-1].copy()
                    next_waypoint[2:] = 0
                else:
                    # print(target_goal_dist, t)
                    next_waypoint = ll_sequence[-2].copy()
            else:
                next_waypoint = ll_sequence[-1].copy()
                next_waypoint[2:] = 0

        state = observation.copy()
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # next_observation, reward, terminal, _ = env_eval.step(action)
        # t += 1
        # total_reward.append(reward)
        # score = env_eval.get_normalized_score(sum(total_reward))

        # Calculate plan_vel and action
        # plan_vel = next_waypoint[:2] - state[:2] if t == 0 else next_waypoint[:2] - sequence[t-1][:2]
        # action = 12.5 * (next_waypoint[:2] - state[:2]) + 1.2 * (plan_vel - state[2:])
        action = np.clip(action, -1, 1)
        next_observation, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        rollout.append(next_observation.copy())

        observation = next_observation
        if terminal:
            break
    print(f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | Time: {planning_time:.2f} | ")
    # plan_rollout = [np.array(plan)[0], np.array(rollout), np.array(rollout[first_time_active:])]
    plan_rollout = [np.array(rollout)]
    renderer.composite(join(hl_args.savepath, f'plan_rollout{i}.png'), plan_rollout, ncol=1)
    # print("save path", hl_args.savepath)
    rollouts.append(rollout)
    total_rewards.append(total_reward)

    ## save result as a json file
    json_path = join(hl_args.savepath, f"idx{i}_rollout.json")
    json_data = {'step': t, 'return': total_reward, 'term': terminal, 'planning_time': planning_time}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    success_rate.append(total_reward > 0)
    
_success_rate = np.sum(success_rate)/n_samples*100
print(f"{np.mean(scores):.1f} +/- {np.std(scores)/np.sqrt(len(scores)):.2f} ({_success_rate:.2f}%)")
print(f"succes rate: {_success_rate:.2f}%")
json_path = join(hl_args.savepath, f"final.json")
json_data = {'mean': np.mean(scores), 'std': np.std(scores)/np.sqrt(len(scores)), 'success_rate': _success_rate, 'scores': scores, 'epoch': hl_diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)