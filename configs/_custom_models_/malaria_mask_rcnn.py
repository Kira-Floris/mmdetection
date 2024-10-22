# mmdetection/configs/my_custom_models/my_faster_rcnn.py

_base_ = '/kaggle/working/mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py'  # Adjust this to your chosen base model path
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

# Modify other settings as needed
# For example, you might want to adjust the learning rate or number of epochs
# Here’s an example modification:
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)  # Adjust the learning rate
lr_config = dict(policy='step', step=[16, 22])  # Adjust the learning rate schedule
total_epochs = 24  # Set total training epochs
