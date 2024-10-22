default_scope = 'mmdet'

_base_ = '/kaggle/working/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'

# Dataset setup
dataset_type = 'CocoDataset'
data_root = '/kaggle/working/data/malaria/'

data = dict(
    train=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/train_annotations.json', 
        img_prefix=data_root + 'train/'),
    val=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/val_annotations.json', 
        img_prefix=data_root + 'val/'),
    test=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/val_annotations.json', 
        img_prefix=data_root + 'val/')
)

# Dataloader settings
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    shuffle=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False)

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

# Learning rate schedule
# Learning rate schedule
param_scheduler = [
    dict(type='StepLR', step=[16, 22], gamma=0.1)  # StepLR with gamma and step
]

# Training schedule
max_epochs = 24
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# RPN and RCNN configurations
train_cfg.update(
    dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    )
)

# Validation configuration
val_cfg = dict(
    metric=['bbox'],
    interval=1
)

# Validation configuration
val_evaluator = dict(
    type='CocoEvaluator',  # Use COCO-style evaluator
    ann_file=data_root + 'annotations/val_annotations.json',
    metric='bbox')

# Testing configuration
test_cfg = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

# Test evaluator configuration
test_evaluator = dict(
    type='CocoEvaluator',
    ann_file=data_root + 'annotations/val_annotations.json',
    metric='bbox')

# Hooks for logging and checkpointing
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))

# Default hooks for logging and checkpointing
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',  # Specify the type of hook
        interval=5,
        max_keep_ckpts=2,
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5)
)

# Visualizer configuration
visualizer = dict(
    type='Visualizer',  # Use 'Visualizer' if it exists, or choose a suitable type
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='malaria_faster_rcnn',  # Name for the visualizer
    save_dir='/kaggle/working/mmdetection/work_dirs/malaria_faster_rcnn'  # Directory to save outputs
)


# Pretrained weights
load_from = './checkpoints/faster_rcnn_r50_fpn_2x_coco.pth'
