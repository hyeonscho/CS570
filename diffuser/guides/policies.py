from collections import namedtuple

# import numpy as np
import torch
import einops
import pdb
import torch.nn as nn
import diffuser.utils as utils
from diffuser.models import helpers
import numpy as np

Trajectories = namedtuple("Trajectories", "actions observations")

def last_value(traj2compare, goal_point):
    dist = -torch.linalg.norm(
        traj2compare[:, -1, 2:] - goal_point[:, :],
        dim=-1) # (b, c') -> (b,)
    return torch.exp(dist)

def every_value(traj2compare, goal_point):
    dist = -torch.linalg.norm(
        traj2compare[..., 2:] - goal_point, dim=-1) # (b, t, c') -> (b, t)
    return torch.exp(dist).sum(-1)

class TrueValueGuide(nn.Module):
    def __init__(self, func_type, horizon):
        super().__init__()
        self.horizon = horizon
        self.func_type = func_type

    def forward(self, x, cond, t):
        if self.func_type == "every":
            goal_point = cond[self.horizon-1]
            goal_point = torch.tile(goal_point[:, None, :], (1, x.shape[1], 1))
            goal_point[..., 2:] = x[..., 4:]
            row_score = every_value(x, goal_point)
        elif self.func_type == "last":
            goal_point = cond[self.horizon-1]
            row_score = last_value(x, goal_point)

        return row_score.squeeze(dim=-1)
    
    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad
    
class TrueValueGuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, n_guide_steps=5, scale=0.1, t_stopgrad=2, scale_grad_by_std=True):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.n_guide_steps = n_guide_steps
        self.scale = scale
        self.t_stopgrad = t_stopgrad
        self.scale_grad_by_std = scale_grad_by_std

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        if batch_size >= 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "d -> repeat d",
                repeat=batch_size,
            )
        return conditions
    
    def n_step_guided_p_sample(self, model, x, cond, t, guide, scale=1.0, t_stopgrad=0, n_guide_steps=100, scale_grad_by_std=True):
        model_log_variance = helpers.extract(model.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
        
        traj_cond = {0: cond[0]}
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                x_g = self.diffusion_model.predict_start_from_noise(x, t, self.diffusion_model.model(x, traj_cond, t))
                y, grad = guide.gradients(x_g, cond, t)
            
            if scale_grad_by_std:
                grad = model_var * grad
            
            grad[t < t_stopgrad] = 0
            
            x = x + scale * grad
            x = helpers.apply_conditioning(x, traj_cond, model.action_dim)
            
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
        
        # no noise when t == 0
        
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        
        return model_mean + model_std * noise, y
    
    def __call__(self, conditions, batch_size=1, **kwargs):
        # breakpoint()

        conditions = self._format_conditions(conditions, batch_size)
        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        # sample = self.diffusion_model(conditions)

        with torch.no_grad():
            device = self.diffusion_model.betas.device
            horizon = self.diffusion_model.horizon
            shape = (batch_size, horizon, self.diffusion_model.transition_dim)
            x = torch.randn(shape, device=device)
            x = helpers.apply_conditioning(x, {0: conditions[0]}, self.diffusion_model.action_dim, self.diffusion_model.observation_dim)
            # if return_statistics:
            #     additional_info = {}
            #     values_in_denoising = []
            #     pred_trajs_for_value_estimations = []
            progress = utils.Progress(self.diffusion_model.n_timesteps)
            for i in reversed(range(0, self.diffusion_model.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, val = self.n_step_guided_p_sample(self.diffusion_model, x, conditions, timesteps, 
                                           self.guide, self.scale, self.t_stopgrad, self.n_guide_steps, self.scale_grad_by_std)
                
                x = helpers.apply_conditioning(x, {0: conditions[0]}, self.diffusion_model.action_dim, self.diffusion_model.observation_dim)
                # if return_statistics:
                #     pred_trajs_for_value_estimations.append(x.detach().cpu())
                #     values_in_denoising.append(val.detach().cpu())
                progress.update({'t': i})
        
            progress.close()

        sample = utils.to_np(x)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, : self.diffusion_model.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        # if return_statistics:
        #     pred_trajs_for_value_estimations = torch.stack([
        #         self.normalizer.unnormalize(pred_traj[..., self.action_dim:],
        #                                     'observations') for pred_traj in pred_trajs_for_value_estimations])
        #     pred_trajs_for_value_estimations = utils.to_np(pred_trajs_for_value_estimations)
        #     additional_info["pred_trajs_for_value_estimation"] = pred_trajs_for_value_estimations
        #     additional_info["values"] = utils.to_np(torch.stack(values_in_denoising))

        trajectories = Trajectories(actions, observations)

        # if return_statistics:
        #     return action, trajectories, additional_info
        # else:
        #     return action, trajectories
        return action, trajectories
    

def every_value_antmaze(traj2compare, goal_point):
    dist = -torch.linalg.norm(
        traj2compare - goal_point, dim=-1) # (b, t, c') -> (b, t)
    return torch.exp(dist).sum(-1)

class TrueValueGuideAntmaze(nn.Module):
    def __init__(self, func_type, horizon):
        super().__init__()
        self.horizon = horizon
        self.func_type = func_type

    def forward(self, x, cond, t):
        if self.func_type == "every":
            goal_point = cond[self.horizon-1]
            goal_point = torch.tile(goal_point[:, None, :], (1, x.shape[1], 1))
            row_score = every_value_antmaze(x, goal_point)
        elif self.func_type == "last":
            breakpoint()
            goal_point = cond[self.horizon-1]
            row_score = last_value(x, goal_point)

        return row_score.squeeze(dim=-1)
    
    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad
    
class TrueValueGuidedAntmazePolicy:

    def __init__(self, guide, diffusion_model, normalizer, n_guide_steps=5, scale=0.1, t_stopgrad=2, scale_grad_by_std=True):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.n_guide_steps = n_guide_steps
        self.scale = scale
        self.t_stopgrad = t_stopgrad
        self.scale_grad_by_std = scale_grad_by_std

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        if batch_size >= 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "d -> repeat d",
                repeat=batch_size,
            )
        return conditions
    
    def n_step_guided_p_sample(self, model, x, cond, t, guide, scale=1.0, t_stopgrad=0, n_guide_steps=100, scale_grad_by_std=True):
        model_log_variance = helpers.extract(model.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
        
        traj_cond = {0: cond[0]}
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                x_g = self.diffusion_model.predict_start_from_noise(x, t, self.diffusion_model.model(x, traj_cond, t))
                y, grad = guide.gradients(x_g, cond, t)
            
            if scale_grad_by_std:
                grad = model_var * grad
            
            grad[t < t_stopgrad] = 0
            
            x = x + scale * grad
            x = helpers.apply_conditioning(x, traj_cond, model.action_dim)
            
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
        
        # no noise when t == 0
        
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        
        return model_mean + model_std * noise, y
    
    def __call__(self, conditions, batch_size=1, **kwargs):
        # breakpoint()

        conditions = self._format_conditions(conditions, batch_size)
        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        # sample = self.diffusion_model(conditions)

        with torch.no_grad():
            device = self.diffusion_model.betas.device
            horizon = self.diffusion_model.horizon
            shape = (batch_size, horizon, self.diffusion_model.observation_dim) #
            x = torch.randn(shape, device=device)
            x = helpers.apply_conditioning(x, {0: conditions[0]}, self.diffusion_model.action_dim, self.diffusion_model.observation_dim)
            # if return_statistics:
            #     additional_info = {}
            #     values_in_denoising = []
            #     pred_trajs_for_value_estimations = []
            progress = utils.Progress(self.diffusion_model.n_timesteps)
            for i in reversed(range(0, self.diffusion_model.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, val = self.n_step_guided_p_sample(self.diffusion_model, x, conditions, timesteps, 
                                           self.guide, self.scale, self.t_stopgrad, self.n_guide_steps, self.scale_grad_by_std)
                
                x = helpers.apply_conditioning(x, {0: conditions[0]}, self.diffusion_model.action_dim, self.diffusion_model.observation_dim)
                # if return_statistics:
                #     pred_trajs_for_value_estimations.append(x.detach().cpu())
                #     values_in_denoising.append(val.detach().cpu())
                progress.update({'t': i})
        
            progress.close()

        sample = utils.to_np(x)

        # ## extract action [ batch_size x horizon x transition_dim ]
        # actions = sample[:, :, : self.diffusion_model.action_dim]
        # actions = self.normalizer.unnormalize(actions, 'actions')
        # # actions = np.tanh(actions)

        # ## extract first action
        # action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, :2]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        # if return_statistics:
        #     pred_trajs_for_value_estimations = torch.stack([
        #         self.normalizer.unnormalize(pred_traj[..., self.action_dim:],
        #                                     'observations') for pred_traj in pred_trajs_for_value_estimations])
        #     pred_trajs_for_value_estimations = utils.to_np(pred_trajs_for_value_estimations)
        #     additional_info["pred_trajs_for_value_estimation"] = pred_trajs_for_value_estimations
        #     additional_info["values"] = utils.to_np(torch.stack(values_in_denoising))

        trajectories = Trajectories(None, observations)

        # if return_statistics:
        #     return action, trajectories, additional_info
        # else:
        #     return action, trajectories
        return None, trajectories

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
