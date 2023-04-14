# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import Tensor

from mmcv.cnn import build_activation_layer
from mmengine.utils import to_2tuple
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import ConfigType

from ..utils import LayerNorm2d


@MODELS.register_module(force=True)
class PromptEncoder(BaseModule):

    def __init__(self,
                 embed_dims: int,
                 image_embed_size: int,
                 image_size: int,
                 mask_channels: int,
                 act_cfg: ConfigType = dict(type='GELU'),
                 init_cfg: Optional[dict] = None) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dims (int): The prompts' embedding dimension
          image_embed_size (int): The spatial size of the
            image embedding, as (H, W).
          image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_channels (int): The number of hidden channels used for
            encoding input masks.
          act_cfg (ConfigType): The activation to use when encoding
            input masks.
        """
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.image_size = to_2tuple(image_size)
        self.image_embed_size = to_2tuple(image_embed_size)
        self.pe_layer = PositionEmbeddingRandom(embed_dims // 2)

        self.num_point_embeds: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dims) for _ in range(self.num_point_embeds)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dims)

        self.mask_input_size = to_2tuple(4 * image_embed_size)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_channels // 4),
            build_activation_layer(act_cfg),
            nn.Conv2d(mask_channels // 4, mask_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_channels),
            build_activation_layer(act_cfg),
            nn.Conv2d(mask_channels, embed_dims, kernel_size=1))
        
        self.no_mask_embed = nn.Embedding(1, embed_dims)

    def get_dense_pe(self) -> Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          Tensor: Positional encoding with shape
            1x(embed_dims)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embed_size).unsqueeze(0)

    def _embed_points(self,
                      points: Tensor,
                      labels: Tensor,
                      pad: bool) -> Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: Tensor) -> Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self,
                        points: Optional[Tuple[Tensor, Tensor]],
                        boxes: Optional[Tensor],
                        masks: Optional[Tensor]) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(self,
                points: Optional[Tuple[Tensor, Tensor]],
                boxes: Optional[Tensor],
                masks: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(Tensor, Tensor) or none): point coordinates
            and labels to embed.
          boxes (Tensor or none): boxes to embed
          masks (Tensor or none): masks to embed

        Returns:
          Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dims), where N is determined by the number of input points
            and boxes.
          Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dims)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dims), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embed_size[0], self.image_embed_size[1])

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, 
                            coords_input: Tensor, 
                            image_size: Tuple[int, int]) -> Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
