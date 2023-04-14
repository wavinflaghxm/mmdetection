# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.structures.mask import mask2bbox
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from ..utils import batch_iterator, calculate_stability_score


@MODELS.register_module(force=True)
class SAMHead(BaseDenseHead):

    def __init__(self,
                 prompt_encoder: ConfigType,
                 mask_decoder: ConfigType,
                 points_per_side: int = 32,
                 points_per_batch: int = 64,
                 multimask_output: bool = True,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)

        self.points_per_batch = points_per_batch
        point_grids = self.get_point_grids(points_per_side)
        self.register_buffer('point_grids', point_grids, persistent=False)

        self.multimask_output = multimask_output
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    @staticmethod
    def get_point_grids(n_per_side: int) -> Tensor:
        offset = 1 / (2 * n_per_side)
        points_one_side = torch.linspace(offset, 1 - offset, n_per_side)
        points_x = torch.tile(points_one_side[None, :], (n_per_side, 1))
        points_y = torch.tile(points_one_side[:, None], (1, n_per_side))
        points = torch.stack([points_x, points_y], axis=-1).reshape(-1, 2)
        return points

    def get_point_inputs(self, batch_data_samples: SampleList) -> SampleList:
        for data_samples in batch_data_samples:
            img_shape = data_samples.img_shape

            points_scale = self.point_grids.new_tensor(img_shape)
            point_coords = self.point_grids * points_scale
            data_samples.point_inputs = point_coords.unsqueeze(1)
            data_samples.point_labels = point_coords.new_ones(
                (point_coords.shape[0], 1), dtype=torch.int32)
        return batch_data_samples
    
    def predict(self,
                x: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_data_samples = self.get_point_inputs(batch_data_samples)

        results_list = []
        for data_samples, curr_embedding in zip(batch_data_samples, x):
            if 'point_inputs' in data_samples:
                points_for_img = (data_samples.point_inputs, 
                                  data_samples.point_labels)
            else:
                points_for_img = None
            boxes_for_img = data_samples.get('box_inputs', None)
            masks_for_img = data_samples.get('mask_inputs', None)
            all_for_img = (*points_for_img, boxes_for_img, masks_for_img)
            cond = [x is not None for x in all_for_img]
            all_for_img = [x for x in all_for_img if x is not None]
            
            results = InstanceData()
            for batches in batch_iterator(self.points_per_batch, *all_for_img):
                points, labels, boxes, masks = [batches.pop(0) if x else None for x in cond]

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(points, labels), boxes=boxes, masks=masks)
                
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=self.multimask_output)
                
                mask_data = self._mask_post_process(
                    low_res_masks,
                    iou_predictions,
                    img_meta=data_samples.metainfo,
                    rescale=rescale)

                results = results.cat([mask_data]) if len(results) == 0 else \
                    results.cat([results, mask_data])

            results.scores = results.iou_preds
            results.labels = results.scores.new_zeros(
                results.scores.shape, dtype=torch.int32)

            if results.bboxes.numel() > 0:
                bboxes = get_box_tensor(results.bboxes)
                det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                    results.labels, self.test_cfg.nms)
                results = results[keep_idxs]
                results.scores = det_bboxes[:, -1]
                if self.test_cfg.get('max_per_img', None) is not None:
                    results = results[:self.test_cfg.max_per_img]
            results_list.append(results)

        return results_list

    def _mask_post_process(self,
                           masks: Tensor,
                           iou_preds: Tensor,
                           img_meta: dict,
                           rescale: bool = True) -> Tensor:
        masks = F.interpolate(
            masks, size=img_meta['img_shape'][:2], mode='bilinear', align_corners=False)
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]

            ori_h, ori_w = img_meta['ori_shape'][:2]
            masks = F.interpolate(
                masks,
                size=[
                    math.ceil(masks.shape[-2] * scale_factor[0]),
                    math.ceil(masks.shape[-1] * scale_factor[1])
                ],
                mode='bilinear',
                align_corners=False)[..., :ori_h, :ori_w]

        mask_threshold = self.test_cfg.get('mask_threshold', 0.)
        pred_iou_thresh = self.test_cfg.get('pred_iou_thresh', 0.)
        stability_score_thresh = self.test_cfg.get('stability_score_thresh', 0.)
        stability_score_offset = self.test_cfg.get('stability_score_offset', 1.0)

        results = InstanceData()
        results.masks = masks.flatten(0, 1)
        results.iou_preds = iou_preds.flatten(0, 1)
        # Filter by predicted IoU
        results = results[results.iou_preds > pred_iou_thresh]
        # Calculate stability score
        results.stability_scores = calculate_stability_score(
            results.masks, 
            mask_threshold, 
            stability_score_offset)
        results = results[results.stability_scores > stability_score_thresh]
        # Threshold masks and calculate boxes
        results.masks = results.masks > mask_threshold
        results.bboxes = mask2bbox(results.masks)

        if self.test_cfg.get('with_mask', False):
            results = results.cpu()
        else:
            delattr(results, 'masks')

        return results
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        raise NotImplementedError

    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head."""
        raise NotImplementedError
