import os
import sys
import torch
import ogbench
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from os.path import join
import pdb
from einops import rearrange
from PIL import Image

from ogbench.pretrain.models.mlp import MLP
from ogbench.pretrain.models.bvae import BetaVAE

from diffuser.guides.policies import TrueValueGuidedVisPolicy, TrueValueGuidedVis
import diffuser.datasets as datasets
import diffuser.utils as utils

import argparse
device='cuda'
class Parser(utils.Parser):
    dataset: str = ''
    config: str = ''

####################################################################################################
################# Helper functions
####################################################################################################
@torch.no_grad()
def decode_position_from_normalized_emb(e2s_model, emb, env_id):
    # Normalizations for medium maze
    pos = e2s_model(emb).detach().cpu().numpy()
    if 'medium' in env_id and 'point' in env_id:
        pos_mean = np.array([10.273524, 9.648321])
        pos_std = np.array([5.627576, 4.897987])
    elif 'large' in env_id and 'point' in env_id:
        pos_mean = np.array([16.702621, 10.974173])
        pos_std = np.array([10.050303, 6.8203936])
    elif 'giant' in env_id and 'point' in env_id:
        pos_mean = np.array([24.888689, 17.158426])
        pos_std = np.array([14.732276, 11.651127])
    pos = pos * pos_std + pos_mean
    pos = pos / 4 + 1
    if not 'giant' in env_id:
        if len(pos.shape) == 2:
            pos = pos[:, [1, 0]]
        else: # if batched
            pos = pos[:, :, [1, 0]]
    return pos
# vae 넣기 전에 pixel을 normalize
def normalize_pixels(obs, env_id): # obs twice, 하나는 Pixel, 다른 하나는 embedding
    # Normalizations for medium maze
    if 'medium' in env_id and 'point' in env_id: # Medium Point maze
        obs_mean = 141.785487953533
        obs_std = 71.0288272312382
    elif 'large' in env_id and 'point' in env_id: # Large Point maze
        obs_mean = 139.58089505083132
        obs_std = 71.31185013523307
    elif 'giant' in env_id and 'point' in env_id: # Giant Point maze
        obs_mean = 141.01873851323037
        obs_std = 73.4250522212486
    obs = (obs - obs_mean) / obs_std
    return obs
def unnormalize_pixels(obs, env_id):
    # Normalizations for medium maze
    if 'medium' in env_id and 'point' in env_id:
        obs_mean = 141.785487953533
        obs_std = 71.0288272312382
    elif 'large' in env_id and 'point' in env_id:
        obs_mean = 139.58089505083132
        obs_std = 71.31185013523307
    elif 'giant' in env_id and 'point' in env_id:
        obs_mean = 141.01873851323037
        obs_std = 73.4250522212486
    obs = obs * obs_std + obs_mean
    return obs
def unnormalize_action(action, env_id):
    if 'medium' in env_id and 'point' in env_id:
        action_mean = torch.tensor([-0.00524961, -0.00168911], dtype=torch.float32, device=action.device)
        action_std = torch.tensor([0.70124096, 0.6971626], dtype=torch.float32, device=action.device)
    elif 'large' in env_id and 'point' in env_id:
        action_mean = torch.tensor([-0.01116096, 0.00125011], dtype=torch.float32, device=action.device)
        action_std = torch.tensor([0.7068106, 0.6878459], dtype=torch.float32, device=action.device)
    elif 'giant' in env_id and 'point' in env_id:
        action_mean = torch.tensor([-0.00714872, -0.00213099], dtype=torch.float32, device=action.device)
        action_std = torch.tensor([0.70283055, 0.69673675], dtype=torch.float32, device=action.device)
    action = action.to(torch.float32)
    action = action * action_std + action_mean
    return action

def unnormalize_emb(emb, env_id):
    if 'medium' in env_id:
        emb_observation_mean =  np.array([0.53332597, -0.57663816, -0.15480594, -0.10989726,  0.13822828, -0.7565398 , -0.67368555, -0.5261524])
        emb_observation_std = np.array([2.230295 , 1.8695153, 2.5765393, 2.5024776, 2.409886 , 2.3264396, 2.2680814, 2.1177504])
    elif 'large' in env_id:
        emb_observation_mean = np.array([0.56942457, -0.7748614, 0.03422518, -0.05451093, 0.00234696, -0.4766496, -0.53628725, -1.1162051])
        emb_observation_std = np.array([2.0956497, 2.2737527, 2.3882532, 2.6977062, 2.1805387, 2.6994274, 2.4300833, 2.137858])

    return emb * emb_observation_std + emb_observation_mean

def normalize_emb(emb, env_id):
    if 'medium' in env_id:
        emb_observation_mean =  np.array([0.53332597, -0.57663816, -0.15480594, -0.10989726,  0.13822828, -0.7565398 , -0.67368555, -0.5261524])
        emb_observation_std = np.array([2.230295 , 1.8695153, 2.5765393, 2.5024776, 2.409886 , 2.3264396, 2.2680814, 2.1177504])
    elif 'large' in env_id:
        emb_observation_mean = np.array([0.56942457, -0.7748614, 0.03422518, -0.05451093, 0.00234696, -0.4766496, -0.53628725, -1.1162051])
        emb_observation_std = np.array([2.0956497, 2.2737527, 2.3882532, 2.6977062, 2.1805387, 2.6994274, 2.4300833, 2.137858])

    return (emb - emb_observation_mean) / emb_observation_std



#---------------------------------- setup ----------------------------------#
n_samples = 10
args = Parser().parse_args('plan', add_extras=True)
args.savepath = args.savepath[:-1] + 'only_for_vis'
save_path = args.savepath
restricted_pd = args.restricted_pd

from diffuser.utils.serialization import mkdir
mkdir(save_path)

env = ogbench.locomaze.maze.make_maze_env(loco_env_type='point',maze_env_type='maze',maze_type=args.dataset.split("-")[-2], ob_type='pixels', render_mode='rgb_array',width=64,height=64, camera_name='back')

env_name = args.dataset
pretrained_model_path = '/home/baek1127/ogbench/embedded_data/'
if 'medium' in env_name:
    pretrained_model_path += 'medium'
elif 'large' in env_name:
    pretrained_model_path += 'large'
elif 'giant' in env_name:
    pretrained_model_path += 'giant'
invd_model_path = pretrained_model_path + '/invd.pth'
e2s_model_path = pretrained_model_path + '/e2s.pth'
vae_model_path = pretrained_model_path + '/vae.pth'
# self.env = load_environment(env)
observation_dim = 8
action_dim = 0

action_model = MLP(observation_dim * 3, 2, hidden_dim=1024, num_layers=3) # use 3 frames
action_model.load_state_dict(torch.load(invd_model_path))
action_model = action_model.to(device)

e2s_model = MLP(observation_dim, 2, hidden_dim=1024, num_layers=4) # position 2
e2s_model.load_state_dict(torch.load(e2s_model_path))
e2s_model = e2s_model.to(device)

vae_model = BetaVAE()
vae_model.load_state_dict(torch.load(vae_model_path))
vae_model = vae_model.to(device)

vae_model.eval()
e2s_model.eval()
action_model.eval()

# env = datasets.load_environment(args.dataset)

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
guide = TrueValueGuidedVis('every', diffusion.horizon)
policy = TrueValueGuidedVisPolicy(guide, diffusion, dataset.normalizer)

# 여기서 horizon이 100이면 총 1000 step을 위해 10번 더 길게 반복
max_planning_steps = 1000 #env.max_episode_steps

scores = []
success_rate = []
replanning_at_every = 50

for i in range(n_samples):
    # env.set_task(task_id = (i % 5) + 1)
    env.set_task(task_id=4)
    observation, info = env.reset()
    goal_obs = info['goal']

    norm_obs = torch.from_numpy(normalize_pixels(observation, env_name)).float()[None, ...] # [64, 64, 3]
    norm_obs = rearrange(norm_obs, "b h w c-> b c h w")
    mu, logvar = vae_model.encode(norm_obs.to(device))
    emb = vae_model.reparameterize(mu, logvar).detach().cpu().numpy()
    emb = normalize_emb(emb, env_name)
    observation_pos = decode_position_from_normalized_emb(e2s_model, torch.tensor(emb).float().to(device), env_name)[0]


    norm_goal = torch.from_numpy(normalize_pixels(goal_obs, env_name)).float()[None, ...] # [64, 64, 3]
    norm_goal = rearrange(norm_goal, "b h w c-> b c h w")
    mu, logvar = vae_model.encode(norm_goal.to(device))
    goal_emb = vae_model.reparameterize(mu, logvar).detach().cpu().numpy()
    goal_emb = normalize_emb(goal_emb, env_name)         
    target = decode_position_from_normalized_emb(e2s_model, torch.tensor(goal_emb).float().to(device), env_name)[0]


    # planning에 필요한 cond
    cond = {diffusion.horizon - 1: np.array(*goal_emb)}


    rollout = [emb.copy()]
    total_reward = 0
    plan = None
    sequence = None
    k = 0

    for t in range(max_planning_steps - 1):
        # if t % diffusion.horizon == 0: # start
        if t % replanning_at_every == 0:
            if k == 0:
                cond[0] = np.array(*emb)
            else:
                cond = {diffusion.horizon - 1: np.array(*goal_emb)}
                cond[0] = rollout[-1][0]
                # for j in range(len(rollout)):
                #     # import ipdb; ipdb.set_trace()
                #     cond[j] = rollout[j][0]

            _, samples = policy(cond, batch_size=args.batch_size)
            # import ipdb; ipdb.set_trace()
            plan = samples.observations
            sequence = plan[0]
            k = k + 1
            cond_pos = 0

        if cond_pos == 0:
            action_model_input = torch.cat([
                torch.tensor(sequence[0][None, ...], dtype=torch.float32).to(device),
                torch.tensor(sequence[0][None, ...], dtype=torch.float32).to(device),
                torch.tensor(sequence[1][None, ...], dtype=torch.float32).to(device),
            ]).reshape(1, -1)
        else:
            action_model_input = torch.cat([
                torch.tensor(sequence[cond_pos-1][None, ...], dtype=torch.float32).to(device),
                torch.tensor(sequence[cond_pos][None, ...], dtype=torch.float32).to(device),
                torch.tensor(sequence[cond_pos+1][None, ...], dtype=torch.float32).to(device),
            ]).reshape(1, -1)
        cond_pos += 1

        action = action_model(action_model_input).detach().cpu().numpy()
        
        action = np.clip(action, -1, 1)
        next_observation, reward, terminal, _, _ = env.step(action)

        # png save
        next_obs_image = Image.fromarray(next_observation)
        next_obs_image.save(join(args.savepath, f"task{i}_{t}.png"))
        # next_obs_ = next_observation.transpose(1, 2, 0)



        next_norm_obs = torch.from_numpy(normalize_pixels(next_observation, env_name)).float()[None, ...] # [1, 64, 64, 3]
        next_norm_obs = rearrange(next_norm_obs, "b h w c-> b c h w")
        mu, logvar = vae_model.encode(next_norm_obs.to(device))
        next_emb = vae_model.reparameterize(mu, logvar).detach().cpu().numpy()
        next_emb = normalize_emb(next_emb, env_name)
        next_observation_pos = decode_position_from_normalized_emb(e2s_model, torch.tensor(next_emb).float().to(device), env_name)[0]

        total_reward += reward
        rollout.append(next_emb.copy())

        if terminal:
            break

        if t > max_planning_steps:
            break

        observation = next_observation

    plan_rollout = [np.array(plan)[0], np.array(rollout).reshape(1, -1, 8)[0]]
    renderer.composite(join(args.savepath, f'plan_rollout{i}.png'), plan_rollout, ncol=2)

    print(f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | ")

    json_path = join(args.savepath, f"idx{i}_rollout.json")
    json_data = {'step': t, 'return': total_reward, 'term': terminal,
                 'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    success_rate.append(total_reward > 0)

_success_rate = np.sum(success_rate)/n_samples*100
print(f"success rate: {_success_rate:.2f}%")
json_path = join(args.savepath, "success_rate.json")
json.dump({'success_rate:': _success_rate}, open(json_path, 'w'))
