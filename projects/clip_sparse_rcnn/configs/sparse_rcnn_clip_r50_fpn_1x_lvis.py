_base_ = './sparse_rcnn_r50_fpn_1x_lvis.py'

custom_imports = dict(
    imports=['projects.clip_sparse_rcnn.models'],
    allow_failed_imports=False)

num_stages = 6
num_proposals = 2000
model = dict(
    rpn_head=dict(num_proposals=num_proposals),
    roi_head=dict(
        type='CLIPSparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        bbox_head=[
            dict(
                type='CLIPDIIHead',
                num_classes=1203,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                clip_cfg=dict(
                    type='ViT-B/32',
                    load_from='data/metadata/lvis_v1_clip_template.npy',
                    ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
                    prompt='all',
                    save_path='data/metadata/lvis_v1_clip_template.npy',
                    cls_alpha=0.4),
                linear_cfg=dict(
                    type='CLIPLinear',
                    use_sigmoid=True,
                    register=False),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            proposal=False)))

evaluation = dict(interval=2, metric=['bbox'])

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
