
_base_ = [
    '/kaggle/working/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '/kaggle/working/mmdetection/configs/_base_/schedules/schedule_1x.py'
]

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/kaggle/working/mmdetection/data/malaria/'

train_ann_file = data_root+'annotations/_annotations.malaria_train.json'
train_data_prefix = 'train/'

val_ann_file = data_root+'annotations/_annotations.malaria_val.json'
val_data_prefix = 'val/'

test_ann_file = 'annotations/_annotations.malaria_test.json'
test_data_prefix = 'test/'

class_name = ('Trophozoite', 'NEG', 'WBC')
num_classes = len(class_name)

metainfo = dict(classes=class_name)

train_batch_size_per_gpu = 2  # Adjust as necessary
train_num_workers = 4
persistent_workers = True

# -----train val related-----
base_lr = 0.02
max_epochs = 2  # Adjust as necessary

model_test_cfg = dict(
    rcnn=dict(
        score_thr=0.1,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300)
)

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height

# -----model related-----
norm_cfg = dict(type='BN')

# -----train val related-----
weight_decay = 0.05

save_checkpoint_intervals = 10
val_interval = 1
max_keep_ckpts = 3

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        # out_channels=256,
        num_classes=num_classes,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            num_classes=num_classes,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]),
            reg_decoded_bbox=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='Pad', size_divisor=16),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # dict(type='PackDetInputs'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='Pad', size_divisor=16),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='Pad', size_divisor=16),
    # No annotations to load, so we skip the annotation-related steps
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        pipeline=train_pipeline
        ))

val_dataloader = dict(
    batch_size=2,  # Set validation batch size
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        pipeline=val_pipeline
        ))

# test_dataloader = test_dataloader = dict(
#     batch_size=2,  # You can adjust the batch size as needed
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     dataset=dict(
#         type='CocoDataset',
#         data_root=data_root,
#         ann_file=test_ann_file,  # Test annotation file (with just image information, no bboxes)
#         data_prefix=dict(img_path=test_data_prefix),
#         pipeline=test_pipeline
#     )
# )

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=val_ann_file,
    metric=['bbox'],
    format_only=False,
    outfile_prefix="/kaggle/working/results/test/",
)

# Add a test evaluator
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2, by_epoch=True)
]

default_scope = 'mmdet'
# load_from = '/kaggle/working/mmdetection/work_dirs/faster_rcnn_malaria/best_ckpt.pth'  # Ensure this path is correct for your environment
auto_scale_lr = dict(base_batch_size=train_batch_size_per_gpu)

train_cfg = dict(max_epochs=max_epochs, val_interval=1)


default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=50),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
    visualization=dict(type='DetVisualizationHook'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    save_dir='/kaggle/working/visualization/'
    )

# visualization=dict( # user visualization of validation and test results
#     type='DetVisualizationHook',
#     draw=False,
#     interval=1,
#     show=False)