# Copyright (c) OpenMMLab. All rights reserved.
import einops
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.utils import to_2tuple
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptMultiConfig

from ..utils import LayerNorm2d, resize_pos_embed


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        x (Tensor): Input tokens with [B, H, W, C].
        window_size (int): Window size.

    Returns:
        Tuple[Tensor, Tuple[Tensor, Tensor]]

        - ``windows``: Windows after partition with
        [B * num_windows, window_size, window_size, C].
        - ``(Hp, Wp)``: Padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: Tensor, window_size: int,
                       pad_hw: Tuple[int, int],
                       hw: Tuple[int, int]) -> Tensor:
    """Window unpartition into original sequences and removing padding.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        x (Tensor): Input tokens with
            [B * num_windows, window_size, window_size, C].
        window_size (int): Window size.
        pad_hw (tuple): Padded height and width (Hp, Wp).
        hw (tuple): Original height and width (H, W) before padding.

    Returns:
        Tensor: Unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """Get relative positional embeddings according to the relative positions
    of query and key sizes.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        q_size (int): Size of query q.
        k_size (int): Size of key k.
        rel_pos (Tensor): Relative position embeddings (L, C).

    Returns:
        Tensor: Extracted positional embeddings according to relative
        positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: Tensor, 
                           q: Tensor, 
                           rel_pos_h: Tensor, 
                           rel_pos_w: Tensor, 
                           q_size: Tuple[int, int], 
                           k_size: Tuple[int, int]) -> Tensor:
    """Borrowed from https://github.com/facebookresearch/segment-anything/

    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn (Tensor): Attention map.
        q (Tensor): Query q in the attention layer with shape
            (B, q_h * q_w, C).
        rel_pos_h (Tensor): Relative position embeddings (Lh, C) for
            height axis.
        rel_pos_w (Tensor): Relative position embeddings (Lw, C) for
            width axis.
        q_size (tuple): Spatial sequence size of query q with (q_h, q_w).
        k_size (tuple): Spatial sequence size of key k with (k_h, k_w).

    Returns:
        Tensor: Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to False.
        input_size (int, optional): Input resolution for calculating the
            relative positional parameter size. Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 use_rel_pos: bool = False,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (input_size is not None), \
                'Input size must be provided if using relative position embed.'
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_embed_dims))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class TransformerEncoderLayer(BaseModule):
    """Encoder layer with window attention in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (ConfigType): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to False.
        window_size (int): Window size for window attention. Defaults to 0.
        input_size (int, optional): Input resolution for calculating the
            relative positional parameter size. Defaults to None.
        init_cfg (ConfigType, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: ConfigType = dict(type='GELU'),
                 norm_cfg: ConfigType = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)[1]

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size))

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        return x


@MODELS.register_module()
class SAMNeck(BaseModule):
    """Channel reduction neck used in SAM.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels. Defaults to 256.
        init_cfg (OptMultiConfig): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 256,
                 freeze: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.sam_neck = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False),
            LayerNorm2d(out_channels, eps=1e-6),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            LayerNorm2d(out_channels, eps=1e-6))
        
        if freeze:
            self.sam_neck.eval()
            for param in self.sam_neck.parameters():
                param.requires_grad = False
    
    def forward(self, x: Sequence[Tensor]) -> Tensor:
        return self.sam_neck(x[-1])


@MODELS.register_module()
class ViTSAM(BaseModule):
    """Vision Transformer as image encoder used in SAM.

    A PyTorch implement of backbone: `Segment Anything
    <https://arxiv.org/abs/2304.02643>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'base', 'large', 'huge'. If use dict, it should have
            below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            - **global_attn_indexes** (int): The index of layers with global
              attention.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        use_abs_pos (bool): Whether to use absolute position embedding.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to True.
        window_size (int): Window size for window attention. Defaults to 14.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        patch_cfg (ConfigType): Configs of patch embeding. 
            Defaults to an empty dict.
        layer_cfgs (MultiConfig): Configs of each transformer layer in encoder.
            Defaults to an empty dict.
        init_cfg (OptMultiConfig): Initialization config dict. Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072,
                'global_attn_indexes': [2, 5, 8, 11]
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096,
                'global_attn_indexes': [5, 11, 17, 23]
            }),
        **dict.fromkeys(
            ['h', 'huge'], {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120,
                'global_attn_indexes': [7, 15, 23, 31]
            }),
    }

    def __init__(self,
                 arch: Union[str, dict] = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_indices: Union[Sequence[int], int] = -1,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = True,
                 window_size: int = 14,
                 norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
                 frozen_stages: int = -1,
                 patch_cfg: ConfigType = dict(),
                 layer_cfgs: MultiConfig = dict(),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels',
                'global_attn_indexes'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.global_attn_indexes = self.arch_settings['global_attn_indexes']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        self.use_abs_pos = use_abs_pos
        if use_abs_pos:
            # Set position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, *self.patch_resolution, self.embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                window_size=window_size if i not in self.global_attn_indexes else 0,
                input_size=self.patch_resolution,
                use_rel_pos=use_rel_pos,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x, patch_resolution = self.patch_embed(x)

        if self.use_abs_pos:
            x = x + resize_pos_embed(
                einops.rearrange(self.pos_embed, 
                                 'b h w c -> b (h w) c'),
                self.patch_resolution,
                patch_resolution,
                num_extra_tokens=0)
            x = self.drop_after_pos(x)

        x = einops.rearrange(x, 'b (h w) c -> b h w c',
                             h=patch_resolution[0])

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self.out_indices:
                x = einops.rearrange(x, 'b h w c -> b c h w')
                outs.append(x)

        return tuple(outs)
