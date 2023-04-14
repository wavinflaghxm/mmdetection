# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_sam(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v       
        if 'image_encoder' in k:
            new_k = k.replace('image_encoder.', 'backbone.')
            if 'blocks' in new_k:
                new_k = new_k.replace('blocks', 'layers')
            if new_k.startswith('backbone.neck'):
                new_k = new_k.replace('backbone.neck', 'neck.0.sam_neck')
            if 'layers' in new_k and 'norm' in new_k:
                new_k = new_k.replace('norm', 'ln')
            if 'mlp' in new_k:
                new_k = new_k.replace('mlp.lin1', 'ffn.layers.0.0')
                new_k = new_k.replace('mlp.lin2', 'ffn.layers.1')
            if 'patch_embed' in k:
                new_k = new_k.replace('proj', 'projection')
            new_ckpt[new_k] = new_v
        else:
            new_k = 'bbox_head.' + k
            if 'mask_decoder' in k:
                if 'norm' in new_k and 'norm_' not in new_k:
                    new_k = new_k.replace('norm', 'ln')
                if 'mlp' in new_k:
                    new_k = new_k.replace('mlp.lin1', 'ffn.layers.0.0')
                    new_k = new_k.replace('mlp.lin2', 'ffn.layers.1')

        new_ckpt[new_k] = new_v
    
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained SAM models to mmdet style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_sam(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
