# Copyright (c) OpenMMLab. All rights reserved.
from .clip_dii_head import CLIPDIIHead
from .clip_linear import CLIPLinear, load_clip_features
from .clip_sparse_roi_head import CLIPSparseRoIHead

__all__ = [
    'CLIPDIIHead', 'CLIPLinear', 'load_clip_features', 'CLIPSparseRoIHead'
]