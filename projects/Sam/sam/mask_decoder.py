# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN

from mmdet.registry import MODELS
from mmdet.utils import ConfigType

from ..utils import LayerNorm2d


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 downsample_rate: int = 1) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.internal_dims = embed_dims // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dims % num_heads == 0, "num_heads must divide embed_dims."

        self.q_proj = nn.Linear(embed_dims, self.internal_dims)
        self.k_proj = nn.Linear(embed_dims, self.internal_dims)
        self.v_proj = nn.Linear(embed_dims, self.internal_dims)
        self.out_proj = nn.Linear(self.internal_dims, embed_dims)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers: (1) self-attention of sparse
    inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
    block on sparse inputs, and (4) cross attention of dense inputs to sparse
    inputs.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        act_cfg (ConfigType): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        skip_first_layer_pe (bool): skip the PE on the first layer.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int = 2048,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 act_cfg: ConfigType = dict(type='ReLU'),
                 norm_cfg: ConfigType = dict(type='LN'),
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = Attention(embed_dims, num_heads)
        self.ln1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.cross_attn_token_to_image = Attention(
            embed_dims, num_heads, downsample_rate=attention_downsample_rate)
        self.ln2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)
        
        self.ln3 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ln4 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.cross_attn_image_to_token = Attention(
            embed_dims, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, 
                queries: Tensor, 
                keys: Tensor, 
                query_pe: Tensor, 
                key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.ln1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.ln2(queries)

        # FFN block
        queries = self.ln3(self.ffn(queries))

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.ln4(keys)

        return queries, keys


@MODELS.register_module(force=True)
class TwoWayTransformer(nn.Module):

    def __init__(self,
                 depth: int,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 act_cfg: ConfigType = dict(type='ReLU'),
                 norm_cfg: ConfigType = dict(type='LN'),
                 attention_downsample_rate: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    num_fcs=num_fcs,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = Attention(
            embed_dims, num_heads,
            downsample_rate=attention_downsample_rate)

        self.norm_final_attn = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self,
                image_embedding: Tensor,
                image_pe: Tensor,
                point_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embed_dims x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embed_dims for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe)

        # Apply the final attenion layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


@MODELS.register_module(force=True)
class MaskDecoder(nn.Module):
    
    def __init__(self,
                 *,
                 transformer_dims: int,
                 transformer: ConfigType,
                 num_multimask_outputs: int = 3,
                 act_cfg: ConfigType = dict(type='GELU'),
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.transformer_dims = transformer_dims
        self.transformer = MODELS.build(transformer)

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dims)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dims)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dims, transformer_dims // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dims // 4),
            build_activation_layer(act_cfg),
            nn.ConvTranspose2d(transformer_dims // 4, transformer_dims // 8, kernel_size=2, stride=2),
            build_activation_layer(act_cfg))
        
        self.output_hypernetworks_mlps = nn.ModuleList()
        for _ in range(self.num_mask_tokens):
            self.output_hypernetworks_mlps.append(
                MLP(transformer_dims, transformer_dims, transformer_dims // 8, 3))

        self.iou_prediction_head = MLP(
            transformer_dims, 
            iou_head_hidden_dim, 
            self.num_mask_tokens, 
            iou_head_depth)

    def forward(self,
                image_embeddings: Tensor,
                image_pe: Tensor,
                sparse_prompt_embeddings: Tensor,
                dense_prompt_embeddings: Tensor,
                multimask_output: bool) -> Tuple[Tensor, Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (Tensor): the embeddings from the image encoder
          image_pe (Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          Tensor: batched predicted masks
          Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(self,
                      image_embeddings: Tensor,
                      image_pe: Tensor,
                      sparse_prompt_embeddings: Tensor,
                      dense_prompt_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    