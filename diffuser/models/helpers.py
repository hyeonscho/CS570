import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, k, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim, k=4):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, k, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim, observation_dim=4):
    # a simple hack -> not efficient but should work -> the dataloader prohibits to have conditions with different keys in each sample -> maybe a problem in the version of torch
    variable_key = conditions.get("key", None)
    variable_val = conditions.get("value", None)
    if variable_key is not None:
        breakpoint()
        variable_key = variable_key[:, 0] # B, H, 1
        for i in range(len(variable_key)):
            t = int(variable_key[i].item())
            x[i, t, action_dim:action_dim+observation_dim] = variable_val[i].clone()

    for t, val in conditions.items():
        if isinstance(t, str):
            continue
        x[:, t, action_dim:action_dim+observation_dim] = val.clone()
    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim, observation_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        
    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        if self.action_dim != 0:
            a0_loss = (
                loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
            ).mean()
            a_loss = (
                loss[:, :, : self.action_dim] / self.weights[:, : self.action_dim]
            ).mean()
            s_loss = (
                loss[:, :, self.action_dim:self.action_dim+self.observation_dim] / self.weights[:, self.action_dim:self.action_dim+self.observation_dim]
            ).mean()
            if loss.shape[2] >= self.action_dim+self.observation_dim:
                l_loss = (
                    loss[:, :, self.action_dim+self.observation_dim:] / self.weights[:, self.action_dim+self.observation_dim:]
                ).mean()
            else:
                l_loss = None
        else:
            a0_loss = torch.zeros(weighted_loss.shape).mean()
            a_loss = torch.zeros(weighted_loss.shape).mean()
            # s_loss = weighted_loss # The two loses are not the same?
            s_loss = (
                loss[:, :, self.action_dim:self.action_dim+self.observation_dim] * self.weights[:, self.action_dim:self.action_dim+self.observation_dim]
            ).mean()
            if loss.shape[2] >= self.action_dim+self.observation_dim:
                l_loss = (
                    loss[:, :, self.action_dim+self.observation_dim:] * self.weights[:, self.action_dim+self.observation_dim:]
                ).mean()
            else:
                l_loss = None
        return weighted_loss, {"a0_loss": a0_loss, "a_loss": a_loss, "s_loss": s_loss, "l_loss": l_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(), utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": corr,
        }

        return loss, info


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}
