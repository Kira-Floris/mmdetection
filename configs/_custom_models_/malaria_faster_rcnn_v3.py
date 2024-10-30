_base_ = [
    '/kaggle/working/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/kaggle/working/mmdetection/configs/_base_/datasets/malaria_detection.py',
    '/kaggle/working/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '/kaggle/working/mmdetection/configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        zero_init_residual=False,
        norm_cfg=norm_cfg,
        init_cfg=None),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg)))
# optimizer
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(_delete_=True, grad_clip=None)
# learning policy
lr_config = dict(warmup_ratio=0.1, step=[65, 71])
runner = dict(type='EpochBasedRunner', max_epochs=73)