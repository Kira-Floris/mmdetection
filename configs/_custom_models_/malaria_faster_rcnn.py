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
    batch_size=2,  # Adjust based on your system's capacity
    num_workers=2,
    shuffle=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    shuffle=False
)

# Modify other settings as needed
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# Learning rate schedule
lr_config = dict(policy='step', step=[16, 22])

# Training schedule
total_epochs = 24

# Training configuration
train_cfg = dict(
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

# Validation configuration
val_cfg = dict()

# Testing configuration
test_cfg = dict()
