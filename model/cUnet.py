# Ref: https://github.com/FutureXiang/ddae

import os
import math
import torch
import torch.nn as nn
from functools import partial


from .common import GroupNormFix, TimeEmbedding, DepthwiseSeparableConv


class ConditionInjection(nn.Module):
    def __init__(self, n_channels, GN_func, size, cond_latent_dim):
        """
        Injects conditional information into the network.
        """
        super().__init__()
                
        # condition projection
        # print(size)
        upsample_t = int(math.log2(size) - 1)
        self.cond_projection = [
            nn.Linear(cond_latent_dim, n_channels*2*2),
            nn.SiLU(),
            nn.Unflatten(1, (n_channels, 2, 2))    # cy这里是2，1；其他系统都是2，2
        ]
        for _ in range(upsample_t):
            self.cond_projection.append(Upsample(n_channels, use_conv=False))
        self.cond_projection = nn.Sequential(*self.cond_projection)
        
        # feature projection
        self.norm = GN_func(n_channels)
        self.feature_projection = nn.Linear(n_channels, n_channels * 2)
        
        # output
        self.output = nn.Linear(n_channels, n_channels)
        
        self.scale = 1 / math.sqrt(math.sqrt(n_channels))
    
    def forward(self, x, cond_vector):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`.
        * `cond` has shape `[batch_size, cond_latent_dim]`.
        """
        batch_size, n_channels, height, width = x.shape
        # print(x.shape)
        # Condition projection
        q = self.cond_projection(cond_vector).view(batch_size, n_channels, -1).permute(0, 2, 1)
        # print(self.cond_projection)
        x_1 = cond_vector
        # for i, layer in enumerate(self.cond_projection):
        #     x_1 = layer(x_1)
        #     print(f"Layer {i}: {layer.__class__.__name__}, output shape: {x_1.shape}")
        # print(q.shape)
        # Feature projection
        h2 = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)
        kv = self.feature_projection(h2)
        k, v = torch.chunk(kv, 2, dim=-1)
        
        # Scaled dot-product attention
        attn = torch.einsum('bid,bjd->bij', q * self.scale, k * self.scale)
        attn = attn.softmax(dim=2)
        out = torch.einsum('bij,bjd->bid', attn, v)
        
        # Reshape and project
        out = out.reshape(batch_size, -1, n_channels)
        # print(out.shape)
        out = self.output(out)
        # print(out.shape)
        out = out.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        
        return (out + x) / math.sqrt(2.)

class ConditionInjection_cy(nn.Module):
    def __init__(self, n_channels, GN_func, size, cond_latent_dim):
        """
        Injects conditional information into the network.
        """
        super().__init__()
                
        # condition projection
        # print(size)
        upsample_t = int(math.log2(size) - 1) + 1
        self.cond_projection = [
            nn.Linear(cond_latent_dim, n_channels*2*1),
            nn.SiLU(),
            nn.Unflatten(1, (n_channels, 2, 1))    # cy这里是2，1；其他系统都是2，2
        ]
        for _ in range(upsample_t):
            self.cond_projection.append(Upsample(n_channels, use_conv=False))
        self.cond_projection = nn.Sequential(*self.cond_projection)
        
        # feature projection
        self.norm = GN_func(n_channels)
        self.feature_projection = nn.Linear(n_channels, n_channels * 2)
        
        # output
        self.output = nn.Linear(n_channels, n_channels)
        
        self.scale = 1 / math.sqrt(math.sqrt(n_channels))
    
    def forward(self, x, cond_vector):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`.
        * `cond` has shape `[batch_size, cond_latent_dim]`.
        """
        batch_size, n_channels, height, width = x.shape
        # print(x.shape)
        # Condition projection
        q = self.cond_projection(cond_vector).view(batch_size, n_channels, -1).permute(0, 2, 1)
        # print(self.cond_projection)
        # print(q.shape)
        # Feature projection
        h2 = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)
        kv = self.feature_projection(h2)
        k, v = torch.chunk(kv, 2, dim=-1)
        
        # Scaled dot-product attention
        attn = torch.einsum('bid,bjd->bij', q * self.scale, k * self.scale)
        attn = attn.softmax(dim=2)
        out = torch.einsum('bij,bjd->bid', attn, v)
        
        # Reshape and project
        out = out.reshape(batch_size, -1, n_channels)
        # print(out.shape)
        out = self.output(out)
        # print(out.shape)
        out = out.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        
        return (out + x) / math.sqrt(2.)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, GN_func, d_k=None):
        """
        Multi-head self-attention block.
        * `n_channels` is the number of input channels.
        * `d_k` is the number of dimensions per head (default: `n_channels`).
        """
        super().__init__()
        d_k = d_k or n_channels
        n_heads = n_channels // d_k

        self.norm = GN_func(n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)  # For q, k, v
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = 1 / math.sqrt(math.sqrt(d_k))
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`.
        """
        batch_size, n_channels, height, width = x.shape

        # Normalize and flatten spatial dimensions
        h = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)

        # Compute q, k, v
        qkv = self.projection(h).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Scaled dot-product attention
        attn = torch.einsum('bihd,bjhd->bijh', q * self.scale, k * self.scale)
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape and project
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return (res + x) / math.sqrt(2)


class Upsample(nn.Module):
    def __init__(self, n_channels, use_conv=True, SepConv=False):
        """
        Upsample by a factor of 2.
        * `use_conv` determines if a convolution is applied after upsampling.
        """
        super().__init__()
        self.use_conv = use_conv
        Conv = DepthwiseSeparableConv if SepConv else nn.Conv2d
        self.conv = Conv(n_channels, n_channels, kernel_size=3, stride=1, padding=1) if use_conv else None

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, n_channels, use_conv=True, SepConv=False):
        """
        Downsample by a factor of 2.
        * `use_conv` determines if a convolution is used (default: True).
        """
        super().__init__()
        self.use_conv = use_conv
        Conv = DepthwiseSeparableConv if SepConv else nn.Conv2d
        self.conv = Conv(n_channels, n_channels, kernel_size=3, stride=2, padding=1) if use_conv else None
        self.pool = nn.AvgPool2d(2) if not use_conv else None

    def forward(self, x):
        return self.conv(x) if self.use_conv else self.pool(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, GN_func, dropout=0.1, SepConv=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `dropout` is the dropout rate
        """
        super().__init__()
        Conv = DepthwiseSeparableConv if SepConv else nn.Conv2d
        
        self.norm1 = GN_func(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = GN_func(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            Conv(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for embeddings
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

    def forward(self, x, t):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h = self.conv1(self.act1(self.norm1(x)))

        # Adaptive Group Normalization
        t_ = self.time_emb(t)[:, :, None, None]
        h = h + t_

        h = self.conv2(self.act2(self.norm2(h)))
        try:
            return (h + self.shortcut(x)) / math.sqrt(2.)
        except:
            print(h.shape, self.shortcut(x).shape)
            import ipdb; ipdb.set_trace()


class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn, attn_channels_per_head, dropout, GN_func, SepConv):
        """
        Residual block with optional attention mechanism.
        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, GN_func, dropout=dropout, SepConv=SepConv)
        self.attn = AttentionBlock(out_channels, GN_func, attn_channels_per_head) if has_attn else nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, attn_channels_per_head, dropout, GN_func, SepConv):
        """
        Middle block with a residual-attention-residual structure.
        """
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, GN_func, dropout=dropout, SepConv=SepConv)
        self.attn = AttentionBlock(n_channels, GN_func, attn_channels_per_head)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, GN_func, dropout=dropout, SepConv=SepConv)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x
 

class cUNet(nn.Module):
    def __init__(self, image_shape=(3, 32, 32), n_channels=128,
                 ch_mults=(1, 2, 2, 2), is_attn=(False, True, False, False),
                 attn_channels_per_head=None, dropout=0.1, cond_latent_dim=256,
                 n_blocks=2, SepConv=False):
        """
        U-Net architecture for image processing tasks.

        Args:
            image_shape: The (channels, height, width) size of input images.
            n_channels: The number of initial feature map channels.
            ch_mults: Multiplier for the number of channels at each resolution.
            is_attn: Boolean flags indicating attention at each resolution.
            attn_channels_per_head: Number of attention channels per head, None means 1 head.
            dropout: Dropout rate.
            n_blocks: Number of blocks at each resolution.
            SepConv: Use separable convolutions instead of standard convolutions.
        """
        super().__init__()
        Conv = DepthwiseSeparableConv if SepConv else nn.Conv2d

        n_resolutions = len(ch_mults)
        
        # Dynamically bind GroupNormFix to the given n_channels
        self.GroupNormFix = partial(GroupNormFix, n_channels)

        # Time embedding
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding(time_channels)
        self.scale_emb = TimeEmbedding(time_channels)
        
        # Projection for image and condition
        self.image_shape = image_shape
        self.image_proj = Conv(image_shape[0], n_channels, kernel_size=3, padding=1)

        # Downsample stages
        down = []
        in_channels = n_channels
        h_channels = [n_channels]

        for i in range(n_resolutions):
            out_channels = n_channels * ch_mults[i]

            # Add blocks at the current resolution
            down.append(ResAttBlock(in_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout, self.GroupNormFix, SepConv=SepConv))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 1):
                down.append(ResAttBlock(out_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout, self.GroupNormFix, SepConv=SepConv))
                h_channels.append(out_channels)

            # Downsample (except for the last resolution)
            if i < n_resolutions - 1:
                down.append(Downsample(out_channels, SepConv=SepConv))
                h_channels.append(out_channels)

            in_channels = out_channels

        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, attn_channels_per_head, dropout, self.GroupNormFix, SepConv=SepConv)

        # Upsample stages
        up = []
        cond, size = [], image_shape[-1]//(2**(n_resolutions-1))
        for i in reversed(range(n_resolutions)):
            out_channels = n_channels * ch_mults[i]

            # Add blocks at the current resolution
            for _ in range(n_blocks + 1):
                cond.append(ConditionInjection(in_channels, self.GroupNormFix, size, cond_latent_dim))
                up.append(ResAttBlock(in_channels + h_channels.pop(), out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout, self.GroupNormFix, SepConv=SepConv))
                in_channels = out_channels

            # Upsample (except for the last resolution)
            if i > 0:
                up.append(Upsample(out_channels, SepConv=SepConv))
                size *= 2

        assert not h_channels
        self.up = nn.ModuleList(up)
        self.cond = nn.ModuleList(cond)

        # Final layers
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.final = Conv(out_channels, image_shape[0], kernel_size=3, padding=1)

    def forward(self, x, t, cond_vector, scale):
        # x: [B*(1+horizon), C, H, W].
        # t: [B*(1+horizon)]
        # cond: [B*(1+horizon), C, H, W]
        # scale: [B*(1+horizon)]
        
        t = self.time_emb(t)
        scale = self.scale_emb(scale)
        
        x = self.image_proj(x)

        # Store outputs for skip connections
        h = [x]

        # Downsample
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            else:
                x = m(x, t).contiguous()
            h.append(x)

        # Middle block
        x = self.middle(x, t).contiguous()

        # Upsample
        count = 0
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Inject conditional information
                x = self.cond[count](x, cond_vector)
                count += 1
                
                # Concatenate with skip connection
                s = h.pop()
                x = torch.cat([x, s], dim=1)
                x = m(x, t).contiguous()

        # Final layer
        return self.final(self.act(self.norm(x)))



# if __name__ == '__main__':
#     net = UNet((2, 64, 64), 64, (1, 2, 2, 2), (True, True, True, True), None, 0.1, 256, 2, False)
#     import torch
#     # x = torch.zeros(4, 3, 32, 32)
#     # t = torch.zeros(4,)
#     # cond = torch.zeros(4, 3, 32, 32)
#     # scale = torch.zeros(4,)
#     # print(net(x, t, cond=cond, scale=scale).shape)

#     for name, param in net.named_parameters():
#         print(name, param.numel() / 1e6)
#     print(sum(p.numel() for p in net.parameters()) / 1e6)