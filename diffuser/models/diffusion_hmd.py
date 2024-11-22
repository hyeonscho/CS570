import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import pdb
import einops

from .diffusion import GaussianDiffusion

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

from diffuser.utils.debug import debug

# Not good
class GaussianDiffusionHMDBase(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        self.level_dim = kwargs.pop("level_dim")
        super().__init__(*args, **kwargs)
        print('level_dim : ', self.level_dim, "\n")
        self.level_layer = nn.Linear(self.level_dim, self.level_dim)
        self.transition_dim += self.level_dim
        
        print("horizon: ", self.horizon, "\n")
        
        level_weight = 1.0
        loss_weights = self._get_loss_weights(kwargs["action_weight"], level_weight, kwargs["loss_discount"], kwargs["loss_weights"])
        
        self.loss_fn = Losses[kwargs["loss_type"]](loss_weights, self.action_dim, self.observation_dim)

    def _get_loss_weights(self, action_weight, level_weight, discount, weights_dict):
        loss_weights = super().get_loss_weights(action_weight, discount, weights_dict)
        loss_weights[:, self.action_dim+self.observation_dim:] = level_weight
        return loss_weights

    def apply_conditioning(self, x, conditions, level, replace_level=True, add_level=False):
        # x: BxTxD for a single denoising step
        ret = super().apply_conditioning(x, conditions)
        if replace_level:
            ret[:,:,self.action_dim+self.observation_dim:] = level
        if add_level:
            ret = torch.cat([ret, level], dim=-1)
        return ret
    
    # cond is not used -> it is passed to p_mean_variance but not used inside the model -> so, why is it passed?
    # def p_mean_variance(self, x, cond, t, level, return_x_recon=False):
    #     pass
    
    # cond is not used -> it is passed to p_mean_variance but not used inside the model -> so, why is it passed?
    # @torch.no_grad()
    # def p_sample(self, x, cond, t):
    #     pass
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, level, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x_sample = x.clone()
        if self.condition:
            # apply_conditioning -> add the conditions at the end and at the begining of the noise
            # As initial and terminal conditions in Control Theory
            x = self.apply_conditioning(x, cond, level)
            x_sample = self.apply_conditioning(x_sample, cond, level)

        if return_diffusion:
            diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, x_sample = self.p_sample(x_sample, cond, timesteps)
            if self.condition:
                x = self.apply_conditioning(x, cond, level, replace_level=False)
                x_sample = self.apply_conditioning(x_sample, cond, level, replace_level=False)

            # progress.update({'t': i})

            if return_diffusion:
                diffusion.append(x_sample)

        # progress.close()
        if return_diffusion:
            return x_sample, torch.stack(diffusion, dim=1)
        else:
            return x


    @torch.no_grad()
    def conditional_sample(self, cond, level, *args, horizon=None, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        device = self.betas.device
        for k, v in cond.items():
            batch_size = cond[k].shape[0]
        # batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        level = self.level_layer(level) # BxD
        # level_repeated = einops.repeat(level, 'd -> b n d', b=batch_size, n=horizon) # repeat
        level_repeated = einops.repeat(level, 'b d -> b n d', n=horizon) # repeat
        # assert len(level_repeated.shape) == len(x.shape)

        return self.p_sample_loop(shape, cond, level_repeated, *args, **kwargs)


    def p_losses(self, x_start, cond, t, level, return_rec=False):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.condition:
            x_noisy = self.apply_conditioning(x_noisy, cond, level)

        x_recon = self.model(x_noisy, cond, t)
        if self.condition:
            x_recon = self.apply_conditioning(x_recon, cond, level, replace_level=False)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        if return_rec:
            return loss, info, x_recon
        else:
            return loss, info


    def loss(
        self,
        x,
        cond,
        level,
        return_rec=False,
        eval_n=None,
    ):
        batch_size = len(x)
        device = x.device
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        level = self.level_layer(level) # BxD
        level_repeated = einops.repeat(level, 'b d -> b n d', n=x.shape[1]) # repeat
        assert len(level_repeated.shape) == len(x.shape)
        x = torch.cat([x, level_repeated], dim=-1)
        return self.p_losses(x, cond, t, level_repeated, return_rec)

    def forward(self, cond, level, *args, **kwargs):
        return self.conditional_sample(cond=cond, level=level, *args, **kwargs)


# TODO: Make another one that conditions the level similar to the time
class GaussianDiffusionHMD2(GaussianDiffusionHMDBase):
    pass

# Good
class GaussianDiffusionHMDNoLevelWeight(GaussianDiffusionHMDBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        level_weight = 0.0
        loss_weights = self._get_loss_weights(kwargs["action_weight"], level_weight, kwargs["loss_discount"], kwargs["loss_weights"])
        self.loss_fn = Losses[kwargs["loss_type"]](loss_weights, self.action_dim, self.observation_dim)
        print(len(loss_weights))


    def apply_conditioning(self, x, conditions, level, replace_level=True, add_level=False):
        # x: BxTxD for a single denoising step
        # always will replace the level
        return super().apply_conditioning(x, conditions, level, replace_level=True, add_level=add_level)
    
# Good
class GaussianDiffusionHMDNoLevelOut(GaussianDiffusionHMDBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        level_weight = 0.0
        loss_weights = self._get_loss_weights(kwargs["action_weight"], level_weight, kwargs["loss_discount"], kwargs["loss_weights"])
        loss_weights = loss_weights[:, :self.action_dim+self.observation_dim]
        self.loss_fn = Losses[kwargs["loss_type"]](loss_weights, self.action_dim, self.observation_dim)
    
    def p_mean_variance(self, x, cond, t, l, return_x_recon=False):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))
        x_recon = torch.cat([x_recon, l], dim=-1)
        
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_x_recon:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, level):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, l=level)
        if self.sample_noise:
            noise = torch.randn_like(x)
        else:
            noise = 0.0 * torch.randn_like(x)
        # no noise when t == 0 -> t=diffusion denoising step -> the final denoised trajectory
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return (
            model_mean,
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
        )
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, level, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x_sample = x.clone()
        if self.condition:
            # apply_conditioning -> add the conditions at the end and at the begining of the noise
            # As initial and terminal conditions in Control Theory
            x = self.apply_conditioning(x, cond, level)
            x_sample = self.apply_conditioning(x_sample, cond, level)

        if return_diffusion:
            diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, x_sample = self.p_sample(x_sample, cond, timesteps, level)
            if self.condition:
                x = self.apply_conditioning(x, cond, level, replace_level=False, add_level=False)
                x_sample = self.apply_conditioning(x_sample, cond, level, replace_level=False, add_level=False)

            # progress.update({'t': i})

            if return_diffusion:
                diffusion.append(x_sample)

        # progress.close()
        if return_diffusion:
            return x_sample, torch.stack(diffusion, dim=1)
        else:
            return x


    def p_losses(self, x_start, cond, t, level, return_rec=False):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.condition:
            x_noisy = self.apply_conditioning(x_noisy, cond, level)

        x_recon = self.model(x_noisy, cond, t)
        if self.condition:
            x_recon = self.apply_conditioning(x_recon, cond, level, replace_level=False)

        shape4assertion = list(noise.shape)
        shape4assertion[-1] -= self.level_dim
        assert shape4assertion == list(x_recon.shape)

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise[:,:, :self.action_dim+self.observation_dim])
        else:
            loss, info = self.loss_fn(x_recon, x_start[:,:, :self.action_dim+self.observation_dim])

        if return_rec:
            return loss, info, x_recon
        else:
            return loss, info


class GaussianDiffusionNoCondition(GaussianDiffusion):
    def loss(
        self,
        x,
        cond,
        level,
        return_rec=False,
        eval_n=None,
    ):
        batch_size = len(x)
        device = x.device
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, return_rec)


# TODO
class GaussianDiffusionLevelAuxTask(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

        self.level_dim = kwargs.pop("level_dim")
        super().__init__(*args, **kwargs)
        # TODO
        self.classifer = nn.Sequential(
            nn.Conv1d(
                inp_channels=self.transition_dim, out_channels=self.level_dim, kernel_size=3, padding=2
            ),
            # TODO
        )

    # TODO:
    def aux_loss(self, x_recon, level):
        # x_recon: BxTxD
        x_recon = x_recon.permute(0, 2, 1).detach() # BxDxT -> channel first
        cls = self.classifer(x_recon)
        loss = nn.CrossEntropyLoss()
        loss = loss(cls, level)
        info = {"aux_loss": loss.mean()}
        return loss, info
    
    def loss(
        self,
        x,
        cond,
        level,
        return_rec=False,
        eval_n=None,
    ):
        batch_size = len(x)
        device = x.device
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        loss, info, x_recon = self.p_losses(x, cond, t, return_rec=True)
        cls_loss, cls_info = self.aux_loss(x_recon, level)
        info = {**info, **cls_info}
        loss = loss + cls_loss
        if return_rec:
            return loss, info, x_recon
        else:
            return loss, info

    def forward(self, cond, level, *args, **kwargs):
        return self.conditional_sample(cond=cond, level=level, *args, **kwargs)

