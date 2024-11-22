import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer, Stat
from .cloud import sync_logs
from collections import defaultdict

from .training import *

class HMDTrainer(Trainer):
    def render_samples(self, batch_size=2, n_samples=2):
        """
        renders samples from (ema) diffusion model
        """
        for i in range(batch_size):
            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, "cuda:0")
            levels = to_device(batch.levels, "cuda:0")

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                "b d -> (repeat b) d",
                repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions, levels)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.model.action_dim:self.model.action_dim+self.model.observation_dim]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:, None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            if self.ema_model.condition:
                normed_observations = np.concatenate(
                    [
                        np.repeat(normed_conditions, n_samples, axis=0),
                        normed_observations,
                    ],
                    axis=1,
                )

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f"sample-{self.step}-{i}.png")
            self.renderer.composite(savepath, observations)
