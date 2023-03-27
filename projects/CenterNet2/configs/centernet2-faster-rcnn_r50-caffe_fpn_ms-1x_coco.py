_base_ = [
    '_base_/centernet2-faster-rcnn_r50-caffe_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00025,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    # Experiments show that there is no need to turn on clip_grad.
    paramwise_cfg=dict(norm_decay_mult=0.))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
