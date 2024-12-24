from collections import namedtuple

# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
import numpy as np

from diffuser.utils.debug import debug as debug_fn
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple("Trajectories", "actions observations")
TrajectoriesLevels = namedtuple("Trajectories", "actions observations levels")

# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')


class Policy:
    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )

        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cuda:0")
        if batch_size >= 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "d -> repeat d",
                repeat=batch_size,
            )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1, **kwargs):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions, **kwargs)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        if self.action_dim != 0:
            actions = sample[:, :, : self.action_dim]
            action = actions[0, 0]
        else:
            actions = None
            action = None
        ## extract first action

        # if debug:
        act_dim = self.action_dim
        obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories

class HMDPolicy:
    def __init__(self, diffusion_model, normalizer, classifier, level_pairs, jumps, short_seq_len):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.obs_dim = diffusion_model.observation_dim
        self.classifier = classifier
        self.level_pairs = level_pairs
        self.jumps = jumps
        self.short_seq_len = short_seq_len

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )

        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cuda:0")
        if batch_size >= 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "d -> repeat d",
                repeat=batch_size,
            )
        return conditions

    def classify_level(self, observation, target):
        norm_obs = self.normalizer.normalize(observation, "observations")
        norm_tgt = self.normalizer.normalize(target, "observations")
        norm_obs = utils.to_torch(norm_obs, dtype=torch.float32, device="cuda:0")
        norm_tgt = utils.to_torch(norm_tgt, dtype=torch.float32, device="cuda:0")
        # B, 2, D
        inp_classifier = torch.cat([norm_obs[None, None], norm_tgt[None, None]], dim=1)
        preds = self.classifier(inp_classifier).softmax(-1).argmax(-1)
        return preds

    def call_diffuser(self, level, conditions, batch_size, **kwargs):
        level = torch.eye(len(self.jumps))[level].to(self.device).repeat(batch_size, 1)
        sample = self.diffusion_model(conditions, level, **kwargs)
        return sample

    def __call__(self, conditions, level, debug=False, batch_size=1, **kwargs):
        conditions = self._format_conditions(conditions, batch_size)
        
        levels = self.level_pairs[level]
        # from diffuser.utils.debug import debug as debug_fn
        # debug_fn()

        if len(levels) > 1:
            # call_two_levels
            hl_samples = self.call_diffuser(levels[-1], conditions, batch_size, **kwargs)
            # hl_samples is normalized
            # samples = utils.to_np(hl_samples)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part
            samples = hl_samples[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part and action part
            
            
            B, M = samples.shape[:2] # 1, 10, D
            # ll_conditions = np.stack([samples[:, :-1], samples[:, 1:]], axis=2)
            ll_conditions = torch.stack([samples[:, :-1], samples[:, 1:]], dim=2)
            ll_conditions = ll_conditions.reshape(B * (M - 1), 2, -1)
            ll_conditions = {
                0: ll_conditions[:, 0],
                self.short_seq_len - 1: ll_conditions[:, -1],
            }

            samples = self.call_diffuser(levels[-2], ll_conditions, batch_size, **kwargs)
            samples = utils.to_np(samples)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part
            # samples = self.normalizer.unnormalize(samples.copy(), 'observations')  

            samples = samples.reshape(B, (M - 1), self.short_seq_len, -1)
            first_first_segment = samples[:, 0, :1] # take the first of the first segment
            rest_other_segments = samples[:, :, 1:].reshape(B, (M - 1) * (self.short_seq_len-1), -1) # take the rest of each other segments except first
            # print(f"samples: {samples.shape}") # 1, 10, 11, D
            # print(f"first_first_segment: {first_first_segment.shape}, rest_other_segments: {rest_other_segments.shape}")
            sample = np.concatenate([first_first_segment, rest_other_segments],axis=1)
            # sample = samples[0]
            # return ll_sequence
        else:
            # call_one_level
            sample = self.call_diffuser(levels[0], conditions, batch_size, **kwargs)
            sample = utils.to_np(sample)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part

        # ## extract action [ batch_size x horizon x transition_dim ]
        # if self.action_dim != 0:
        #     actions = sample[:, :, : self.action_dim]
        #     action = actions[0, 0]
        # else:
        #     actions = None
        #     action = None
        actions = None
        action = None
        ## extract first action

        # if debug:
        act_dim = self.action_dim
        obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample #[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = TrajectoriesLevels(actions, observations, levels)
        return action, trajectories


# The following class implements the multiscale policy with multi high-levels and the low-level is J1 w/ different conditioning resolution
class HMDPolicyMultiscale2(HMDPolicy):
    def __call__(self, conditions, level, debug=False, batch_size=1, **kwargs):
        conditions = self._format_conditions(conditions, batch_size)
        
        levels = self.level_pairs[level]
        # from diffuser.utils.debug import debug as debug_fn
        # debug_fn()

        if len(levels) > 1:
            # call_two_levels
            hl_samples = self.call_diffuser(levels[-1], conditions, batch_size, **kwargs)
            # hl_samples is normalized
            # samples = utils.to_np(hl_samples)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part
            samples = hl_samples[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part and action part
            
            
            B, M = samples.shape[:2] # 1, 10, D
            # ll_conditions = np.stack([samples[:, :-1], samples[:, 1:]], axis=2)
            ll_conditions = torch.stack([samples[:, :-1], samples[:, 1:]], dim=2)
            ll_conditions = ll_conditions.reshape(B * (M - 1), 2, -1)
            ll_conditions = {
                0: ll_conditions[:, 0],
                self.short_seq_len - 1: ll_conditions[:, -1],
            }

            samples = self.call_diffuser(levels[-2], ll_conditions, batch_size, **kwargs)
            samples = utils.to_np(samples)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part
            # samples = self.normalizer.unnormalize(samples.copy(), 'observations')  

            samples = samples.reshape(B, (M - 1), self.short_seq_len, -1)
            first_first_segment = samples[:, 0, :1] # take the first of the first segment
            rest_other_segments = samples[:, :, 1:].reshape(B, (M - 1) * (self.short_seq_len-1), -1) # take the rest of each other segments except first
            # TODO:
            # cut the trajectories into short_seq_len=self.jumps[levels[-1]] lengthts
            
            # print(f"samples: {samples.shape}") # 1, 10, 11, D
            # print(f"first_first_segment: {first_first_segment.shape}, rest_other_segments: {rest_other_segments.shape}")
            sample = np.concatenate([first_first_segment, rest_other_segments],axis=1)
            # sample = samples[0]
            # return ll_sequence
        else:
            # call_one_level
            sample = self.call_diffuser(levels[0], conditions, batch_size, **kwargs)
            sample = utils.to_np(sample)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part

        # ## extract action [ batch_size x horizon x transition_dim ]
        # if self.action_dim != 0:
        #     actions = sample[:, :, : self.action_dim]
        #     action = actions[0, 0]
        # else:
        #     actions = None
        #     action = None
        actions = None
        action = None
        ## extract first action

        # if debug:
        act_dim = self.action_dim
        obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample #[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = TrajectoriesLevels(actions, observations, levels)
        return action, trajectories

# HMD w/ learanble level token
class HMDPolicy2:
    def __init__(self, diffusion_model, normalizer, level_pairs, jumps, short_seq_len):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.obs_dim = diffusion_model.observation_dim
        self.level_pairs = level_pairs
        self.jumps = jumps
        self.short_seq_len = short_seq_len

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )

        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cuda:0")
        if batch_size >= 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "d -> repeat d",
                repeat=batch_size,
            )
        return conditions

    # def classify_level(self, observation, target):
    #     norm_obs = self.normalizer.normalize(observation, "observations")
    #     norm_tgt = self.normalizer.normalize(target, "observations")
    #     norm_obs = utils.to_torch(norm_obs, dtype=torch.float32, device="cuda:0")
    #     norm_tgt = utils.to_torch(norm_tgt, dtype=torch.float32, device="cuda:0")
    #     # B, 2, D
    #     inp_classifier = torch.cat([norm_obs[None, None], norm_tgt[None, None]], dim=1)
    #     preds = self.classifier(inp_classifier).softmax(-1).argmax(-1)
    #     return preds

    def call_diffuser(self, conditions, batch_size, **kwargs):
        sample = self.diffusion_model(conditions, **kwargs)
        return sample

    def __call__(self, conditions, debug=False, batch_size=1, **kwargs):
        conditions = self._format_conditions(conditions, batch_size)
        
        sample = self.call_diffuser(conditions, batch_size, **kwargs)
        sample = utils.to_np(sample)[:, :, self.action_dim:self.action_dim+self.obs_dim] # remove the level predicition part


        # ## extract action [ batch_size x horizon x transition_dim ]
        # if self.action_dim != 0:
        #     actions = sample[:, :, : self.action_dim]
        #     action = actions[0, 0]
        # else:
        #     actions = None
        #     action = None
        actions = None
        action = None
        ## extract first action

        # if debug:
        act_dim = self.action_dim
        obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample#[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = TrajectoriesLevels(actions, observations, 0)
        return action, trajectories


"""
class Policy:
    def __init__(
        self,
        diffusion_model,
        normalizer,
        jump=1,
        jump_action=False,
        fourier_feature=False,
    ):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.jump = jump
        self.jump_action = jump_action
        self.fourier_feature = fourier_feature

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )

        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cuda:0")
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            "d -> repeat d",
            repeat=batch_size,
        )
        return conditions

    def __call__(
        self, conditions, debug=False, batch_size=1, hd=False, hl=False, **kwargs
    ):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        if hd:
            hl_sample, ll_sample = self.diffusion_model(conditions, **kwargs)
            if hl:
                sample = hl_sample
            else:
                sample = ll_sample
        else:
            sample = self.diffusion_model(conditions, **kwargs)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, : self.action_dim]
        shape = actions.shape
        if self.jump_action:
            actions = self.normalizer.unnormalize(
                actions.reshape(*shape[:-1], 1, -1), "actions"
            )
        else:
            actions = self.normalizer.unnormalize(
                actions.reshape(*shape[:-1], self.jump, -1), "actions"
            )
        actions = actions.reshape(*shape[:-1], -1)
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        if hd:
            if hl:
                act_dim = self.action_dim * 15
                obs_dim = self.diffusion_model.hl_diffuser.observation_dim
                if self.fourier_feature:
                    obs_dim = obs_dim // 3
            else:
                act_dim = self.action_dim
                obs_dim = self.diffusion_model.hl_diffuser.observation_dim
                if self.fourier_feature:
                    obs_dim = obs_dim // 3
        else:
            act_dim = self.action_dim
            obs_dim = self.diffusion_model.observation_dim
        normed_observations = sample[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
        # else:
        #     return action

    def sample_with_context(
        self,
        target_obs,
        ctxt,
        hl_ctxt_len,
        jump,
        ll_ctxt_len=0,
        debug=False,
        batch_size=1,
        hd=False,
        hl=False,
        **kwargs
    ):
        if len(ctxt) <= jump:
            hl_ctxt = ctxt[-1:]
        elif len(ctxt) <= hl_ctxt_len:
            hl_ctxt = ctxt[::-1][::jump][::-1]
        else:
            hl_ctxt = ctxt[-hl_ctxt_len + 1 :]
            hl_ctxt = hl_ctxt[::jump]

        hl_cond = dict()
        for i in range(len(hl_ctxt)):
            hl_cond[i] = hl_ctxt[i]
        hl_cond[self.diffusion_model.hl_diffuser.horizon - 1] = target_obs

        hl_cond = self._format_conditions(hl_cond, batch_size)

        hl_samples = self.diffusion_model.hl_diffuser(cond=hl_cond, **kwargs)
        hl_state = hl_samples[
            :, :, self.diffusion_model.hl_diffuser.action_dim :
        ]  # B, M, C
        B, _ = hl_state.shape[:2]

        ll_cond = {0: hl_state[:1, len(hl_ctxt) - 1], jump: hl_state[:1, len(hl_ctxt)]}

        ll_samples = self.diffusion_model.ll_diffuser(cond=ll_cond, **kwargs)
        ll_samples_ = ll_samples.reshape(B, 1, jump + 1, -1)
        ll_samples = torch.cat(
            [ll_samples_[:, 0, :1], ll_samples_[:, :, 1:].reshape(B, jump, -1)], dim=1
        )

        ## run reverse diffusion process
        if hl:
            sample = hl_samples
        else:
            sample = ll_samples
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, : self.action_dim]
        shape = actions.shape
        if self.jump_action:
            actions = self.normalizer.unnormalize(
                actions.reshape(*shape[:-1], 1, -1), "actions"
            )
        else:
            actions = self.normalizer.unnormalize(
                actions.reshape(*shape[:-1], self.jump, -1), "actions"
            )
        actions = actions.reshape(*shape[:-1], -1)
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        if hl:
            act_dim = self.action_dim * 15
            obs_dim = self.diffusion_model.hl_diffuser.observation_dim
            if self.fourier_feature:
                obs_dim = obs_dim // 3
        else:
            act_dim = self.action_dim
            obs_dim = self.diffusion_model.hl_diffuser.observation_dim
            if self.fourier_feature:
                obs_dim = obs_dim // 3
        normed_observations = sample[:, :, act_dim : act_dim + obs_dim]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
"""
