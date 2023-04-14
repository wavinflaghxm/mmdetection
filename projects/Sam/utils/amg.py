# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Generator, List

import torch
from torch import Tensor


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), \
        "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: Tensor, 
                              mask_threshold: float, 
                              threshold_offset: float) -> Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecesary cast to torch.int64
    intersections = ((masks > (mask_threshold + threshold_offset))
                     .sum(-1, dtype=torch.int16)
                     .sum(-1, dtype=torch.int32))
    unions = ((masks > (mask_threshold - threshold_offset))
              .sum(-1, dtype=torch.int16)
              .sum(-1, dtype=torch.int32))
    
    return intersections / unions
