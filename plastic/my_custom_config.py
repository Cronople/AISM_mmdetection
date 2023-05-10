# the new config inherits the base configs to highlight the necessary modification, config 상속
_base_ = 'C:/Users/PJH/Desktop/AISM_mmdetection/plastic/cascade-mask-rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('PET', 'PS', 'PP', 'PE')

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4)))

# Modify dataset related settings
data_root = './data/coco/'
metainfo = {
    'classes': ('PET', 'PS', 'PP', 'PE'),
    'palette': [
        (220, 20, 60),(200, 0, 0), (150, 60, 30), (200, 200, 50)
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/anno_coco_train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/anno_coco_val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/anno_coco_val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# python tools/train.py configs/plastic/my_custom_config.py