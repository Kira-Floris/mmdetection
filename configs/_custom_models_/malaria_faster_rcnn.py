_base_ = ['../_base_/default_runtime.py', '../_base_/det_faster_rcnn.py']

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/kaggle/working/mmdetection/data/malaria/'

train_ann_file = 'annotations/train_annotations.json'
train_data_prefix = 'train/'

val_ann_file = 'annotations/val_annotations.json'
val_data_prefix = 'val/'

class_name = ('Trophozoite', 'NEG', 'WBC')
num_classes = len(class_name)

metainfo = dict(classes=class_name)

train_batch_size_per_gpu = 2  # Adjust as necessary
train_num_workers = 4
persistent_workers = True

# -----train val related-----
base_lr = 0.002
max_epochs = 20  # Adjust as necessary
num_epochs_stage2 = 10

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    nms_pre=1000,
    score_thr=0.05,  # Threshold to filter out boxes
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (1333, 800)
val_batch_size_per_gpu = 2
val_num_workers = 4

# -----model related-----
norm_cfg = dict(type='BN')

# -----train val related-----
weight_decay = 0.0001

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 5
val_interval_stage2 = 1
max_keep_ckpts = 3
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, gt_max_assign_all=False),
        sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=False),
        pos_weight=-1,
        debug=False),
    test_cfg=model_test_cfg,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        img_prefix=train_data_prefix,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        img_prefix=val_data_prefix,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay))

# learning rate schedule
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2)
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
