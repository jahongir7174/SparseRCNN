# model settings
base = 'https://github.com/SwinTransformer/storage/releases'
pretrained = f'{base}/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(type='SparseMaskRCNN',
             backbone=dict(type='SwinTransformer',
                           embed_dims=96,
                           depths=[2, 2, 6, 2],
                           convert_weights=True,
                           num_heads=[3, 6, 12, 24],
                           init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
             neck=dict(type='PAFPN',
                       in_channels=[96, 192, 384, 768],
                       out_channels=256, num_outs=4,
                       conv_cfg=dict(type='ConvWS'),
                       norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
             rpn_head=dict(type='SparseRPNHead', num_proposals=100),
             roi_head=dict(type='SparseMaskRoIHead',
                           bbox_head=[dict(type='DIIHead',
                                           num_classes=80,
                                           loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                                           loss_cls=dict(type='PolyLoss'),  # nets/nn.py
                                           bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                                           clip_border=False,
                                                           target_means=[0., 0., 0., 0.],
                                                           target_stds=[0.5, 0.5, 1., 1.])) for _ in range(6)],
                           mask_head=[dict(type='DynamicMaskHead',
                                           num_classes=80,
                                           conv_cfg=dict(type='ConvWS'),
                                           norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                                           loss_mask=dict(type='DiceLoss',
                                                          loss_weight=8.0,
                                                          activate=False, eps=1e-5)) for _ in range(6)]),
             # model training and testing settings
             train_cfg=dict(rpn=None,
                            rcnn=[dict(assigner=dict(type='HungarianAssigner',
                                                     cls_cost=dict(type='FocalLossCost', weight=2.0),
                                                     reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                                                     iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
                                       sampler=dict(type='PseudoSampler'),
                                       pos_weight=1,
                                       mask_size=28) for _ in range(6)]),
             test_cfg=dict(rpn=None,
                           rcnn=dict(max_per_img=100, mask_thr_binary=0.5)))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../Dataset/COCO/'
samples_per_gpu = 4
workers_per_gpu = 4
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [dict(type='LoadImageFromFile', to_float32=False, color_type='color'),
                  dict(type='LoadAnnotations', with_mask=True, poly2mask=False),
                  dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                  dict(type='RandomFlip', flip_ratio=0.5),
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='Pad', size_divisor=32),
                  dict(type='DefaultFormatBundle'),
                  dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])]
test_pipeline = [dict(type='LoadImageFromFile'),
                 dict(type='MultiScaleFlipAug',
                      flip=False,
                      img_scale=(1333, 800),
                      transforms=[dict(type='Resize', keep_ratio=True),
                                  dict(type='RandomFlip'),
                                  dict(type='Normalize', **img_norm_cfg),
                                  dict(type='Pad', size_divisor=32),
                                  dict(type='ImageToTensor', keys=['img']),
                                  dict(type='Collect', keys=['img'])])]
data = dict(samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            train=dict(type=dataset_type,
                       ann_file=data_root + 'annotation/train2017.json',
                       img_prefix=data_root + 'images/train2017/',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     ann_file=data_root + 'annotation/val2017.json',
                     img_prefix=data_root + 'images/val2017/',
                     pipeline=test_pipeline),
            test=dict(type=dataset_type,
                      ann_file=data_root + 'annotation/val2017.json',
                      img_prefix=data_root + 'images/val2017/',
                      pipeline=test_pipeline))
optimizer = dict(type='AdamW',
                 lr=0.0001, weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(step=[9, 11],
                 policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
evaluation = dict(interval=12, metric=['bbox', 'segm'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=16)
