import math
import torch
import torch
import torch.nn as nn


def GroupNormFix(num_groups, channels):
    """Returns a GroupNorm layer with Fix groups."""
    return nn.GroupNorm(num_groups, channels)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding.
        """
        super().__init__()
        self.n_channels = n_channels

        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t):
        """
        Compute the time embedding.
        * `t` is the input tensor of shape `[batch_size]`.
        """
        # Create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb_scale = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        # Pass through the MLP
        emb = self.act(self.lin1(emb))
        return self.lin2(emb)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x