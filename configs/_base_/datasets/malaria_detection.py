# configs/_base_/datasets/malaria_dataset.py
from mmcv import Config

cfg = Config({
    'type': 'CocoDataset',
    'ann_file': '/kaggle/working/mmdetection/data/malaria/annotations/train_annotations.json',
    'img_prefix': '/kaggle/working/mmdetection/data/malaria/train/',
    'pipeline': [
        # Define your data preprocessing pipeline here
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', 
             mean=[123.675, 116.28, 103.53], 
             std=[58.395, 57.12, 57.375], 
             to_bgr=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ],
})

cfg_val = Config({
    'type': 'CocoDataset',
    'ann_file': '/kaggle/working/mmdetection/data/malaria/annotations/val_annotations.json',
    'img_prefix': '/kaggle/working/mmdetection/data/malaria/val/',
    'pipeline': [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='Normalize', 
             mean=[123.675, 116.28, 103.53], 
             std=[58.395, 57.12, 57.375], 
             to_bgr=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ],
})
