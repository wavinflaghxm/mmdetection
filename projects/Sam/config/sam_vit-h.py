_base_ = [
    'mmdet::_base_/datasets/lvis_v1_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.Sam.sam'], allow_failed_imports=False)

backbone_norm_cfg = dict(type='LN', eps=1e-6)

vit_embed_dims = 1280
prompt_embed_dims = 256
image_size = 1024
vit_patch_size = 16
image_embed_size = image_size // vit_patch_size

model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ViTSAM',
        arch='huge',
        img_size=image_size,
        patch_size=vit_patch_size,
        window_size=14,
        qkv_bias=True,
        use_rel_pos=True,
        norm_cfg=backbone_norm_cfg),
    neck=[
        dict(type='SAMNeck', 
             in_channels=vit_embed_dims,
             out_channels=prompt_embed_dims,
             freeze=True)
    ],
    bbox_head=dict(
        type='SAMHead',
        points_per_side=32,
        points_per_batch=64,
        prompt_encoder=dict(
            type='PromptEncoder',
            embed_dims=prompt_embed_dims,
            image_embed_size=image_embed_size,
            image_size=image_size,
            mask_channels=16),
        mask_decoder=dict(
            type='MaskDecoder',
            num_multimask_outputs=3,
            transformer=dict(
                type='TwoWayTransformer',
                depth=2,
                embed_dims=prompt_embed_dims,
                feedforward_channels=2048,
                num_heads=8),
            transformer_dims=prompt_embed_dims,
            iou_head_depth=3,
            iou_head_hidden_dim=256)),
    test_cfg=dict(
        with_mask=False,
        mask_threshold=.0,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=1000))

load_from = 'data/pretrained/sam/sam_vit_h_4b8939.pth'

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=True),
    dict(type='Pad', size=(image_size, image_size), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_ann_file = 'lvis/lvis_v1_val_rare_filter_image.json'
# test_ann_file = 'annotations/lvis_v1_val.json'
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

test_evaluator = dict(
    type='LVISMetric',
    ann_file=data_root + test_ann_file,
    metric=['proposal_fast'])

default_hooks = dict(logger=dict(interval=5))
