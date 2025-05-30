import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from .temporal import *

# TODO: integrate linear attention instead of convolutions in the temporal unet
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)

# TODO: Implement this
class TemporalUnetAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
