_base_ = '/kaggle/working/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'

# Dataset setup
dataset_type = 'CocoDataset'
data_root = '/kaggle/working/data/malaria/'

data = dict(
    train=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/train_annotations.json', 
        img_prefix=data_root + 'train/'
    ),
    val=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/val_annotations.json', 
        img_prefix=data_root + 'val/'
    ),
    test=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/val_annotations.json', 
        img_prefix=data_root + 'val/'
    )
)

# Dataloader settings
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    shuffle=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False
)

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        bias_lr_mult=2., bias_decay_mult=0.)
)

# Learning rate schedule
lr_config = dict(policy='step', step=[16, 22])

# Total training epochs
total_epochs = 24

# Model configuration (rpn and rcnn go here)
model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,  # Adjust according to the number of classes
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    # Training settings (train_cfg moved here)
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    )
)

# Validation and Test configurations
val_cfg = dict(interval=1, metric='bbox')

val_evaluator = dict(
    type='CocoEvaluator',
    ann_file=data_root + 'annotations/val_annotations.json',
    metric='bbox'
)

test_cfg = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

test_evaluator = dict(
    type='CocoEvaluator',
    ann_file=data_root + 'annotations/val_annotations.json',
    metric='bbox'
)
