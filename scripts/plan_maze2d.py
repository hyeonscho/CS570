import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# exit()


import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import argparse

parser = argparse.ArgumentParser("diffuser", add_help=False)
parser.add_argument("--cfg", type=str)
parser.add_argument('--rpd', action='store_true')
args, leftovers = parser.parse_known_args()

cfg = args.cfg
# I do not know why the parser does not work -> quick solution -> argparse and override
restricted_pd = args.rpd


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    # config: str = 'config.maze2d_hmdConfig300'
    config: str = cfg
    restricted_pd: bool = restricted_pd


#---------------------------------- setup ----------------------------------#
n_samples = 100

args = Parser().parse_args('plan', add_extras=False) # Discovered it later, so had to disable it as I am doing it here, until I test it

# bad workaround, just for now
args.restricted_pd =  restricted_pd
args.savepath = args.savepath.replace('rpdFalse', f'rpd{restricted_pd}')
args.savepath = args.savepath.replace('rpdTrue', f'rpd{restricted_pd}')


from diffuser.utils.serialization import mkdir
print(args.savepath)
mkdir(args.savepath)
# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

print(args.logbase, args.dataset, args.diffusion_loadpath, args.diffusion_epoch)

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

print(f"{args.diffusion_epoch}")
print(args.savepath)
scores = []
#---------------------------------- main loop ----------------------------------#
for i in range(n_samples):

    observation = env.reset()

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    distance_threshold = 2
    plan = None
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation
            # from diffuser.utils.debug import debug
            # debug()
            _, samples = policy(cond, batch_size=args.batch_size)
            plan = samples.observations
            # actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            # next_waypoint = sequence[-1].copy()
            # next_waypoint[2:] = 0
            
            if restricted_pd:
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

            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            # print(
            #     f'maze | pos: {xy} | goal: {goal}'
            # )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f'{t}.png')

            # if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation
    plan_rollout = [np.array(plan)[0], np.array(rollout)]
    renderer.composite(join(args.savepath, f'plan_rollout{i}.png'), plan_rollout, ncol=2)

    # logger.finish(t, env.max_episode_steps, score=score, value=0)
    print(
        f" {i} / {n_samples}\t t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
    )

    ## save result as a json file
    json_path = join(args.savepath, f"idx{i}_rollout.json")
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    scores.append(score*100)

print(f"{np.mean(scores):.1f} +/- {np.std(scores)/np.sqrt(len(scores)):.2f}")