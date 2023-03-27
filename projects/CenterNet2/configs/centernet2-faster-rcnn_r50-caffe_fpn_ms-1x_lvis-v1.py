_base_ = [
    '_base_/centernet2-faster-rcnn_r50-caffe_fpn.py',
    'mmdet::_base_/datasets/lvis_v1_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py', 
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1203)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

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


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                'project': 'mmdetection',
                'group': 'maskrcnn-r50-fpn-1x-coco'
             },
             interval=50,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)
        ])