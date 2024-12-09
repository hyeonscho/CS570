import os
import sys

sys.path.append('/root/diffuser_chain_hd')
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


class Parser(utils.Parser):
    dataset: str = 'maze2d-xxlarge-v1'
    config: str = 'config.maze2d_390_actionWeight1'


#---------------------------------- setup ----------------------------------#
n_samples = 100

args = Parser().parse_args('diffusion', add_extras=False) # Discovered it later, so had to disable it as I am doing it here, until I test it


env = datasets.load_environment(args.dataset)


dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,#1000,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    jump=args.jump,
    jump_action=args.jump_action,
)


render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()



trjs = []
for i in range(len(dataset)):
    idx = i
    trj = dataset.normalizer.unnormalize(dataset[idx].trajectories, "observations")[:, :2]
    # print(np.linalg.norm(trj - np.array(env._target)))
    # print(np.linalg.norm(trj - np.array(env._target), axis=1).min())
    if (np.linalg.norm(trj - np.array(env._target), axis=1) < 1).any():
        trjs.append(trj)
    # if i > 1000:
    #     break

print(np.array(trjs).shape)
trjs = np.random.choice(trjs, 100)
renderer.composite('/root/diffuser_chain_hd/scripts/trjs.png', np.array(trjs), ncol=10)