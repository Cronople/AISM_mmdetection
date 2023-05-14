# ---------- Config file inheritence------------------
# 
_base_ = './cascade-mask-rcnn_r50_fpn.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
data_root = 'C:/Users/PJH/Desktop/AISM_mmdetection/data/coco/'
backend_args = None


train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile', backend_args=backend_args),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=False,  # Whether to use instance mask, True for instance segmentation
        poly2mask=False),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(
        type='Resize',  # Pipeline that resizes the images and their annotations
        scale=(512, 512),  # The largest scale of the images
        keep_ratio=True  # Whether to keep the ratio between height and width
        ),
    dict(
        type='RandomFlip',  # Augmentation pipeline that flips the images and their annotations
        prob=0.5),  # The probability to flip
    dict(type='PackDetInputs')  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]
test_pipeline = [  # Testing data processing pipeline
    dict(type='LoadImageFromFile', backend_args=backend_args),  # First pipeline to load images from file path
    dict(type='Resize', scale=(512, 512), keep_ratio=True),  # Pipeline that resizes the images
    dict(
        type='PackDetInputs',  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=2,   # Batch size of a single GPU
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,    #If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(  # training data sampler
    type='DefaultSampler', shuffle=True), # DefaultSampler which supports both distributed and non-distributed training. Refer to https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Batch sampler for grouping images with similar aspect ratio into a same batch. It can reduce GPU memory cost.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotation_data/ann_file.json',
        data_prefix=dict(img='train/image_data/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # Config of filtering images and annotations
        pipeline=train_pipeline,
        backend_args=backend_args # metainfo=metainfo 제거했음
        ))
val_dataloader = dict(
    batch_size=1,   # Batch size of a single GPU. If batch-size > 1, the extra padding area may influence the performance.
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    drop_last=False,    # # Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size
    sampler=dict(
        type='DefaultSampler', shuffle=False),   # not shuffle during validation and testing
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='val/annotation_data/ann_file.json',
            data_prefix=dict(img='val/image_data/'),
            test_mode=True,   # Turn on the test mode of the dataset to avoid filtering annotations or images
            pipeline=test_pipeline,
            backend_args=backend_args))
test_dataloader = val_dataloader    # Testing dataloader config


# Modify metric related settings
val_evaluator = dict(
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + 'val/annotation_data/ann_file.json',  # Annotation file path
    metric=['bbox', 'segm'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator  # Testing evaluator config

# ----------test 데이터셋이 annotaion file이 있다면 다음 코드 적용-------------
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     metric=['bbox', 'segm'],  # Metrics to be evaluated
#     format_only=True,  # Only format and save the results to coco json file
#     outfile_prefix='./work_dirs/coco_detection/test')  # The prefix of output json files

# Training and testing config
train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type

# Optimization config
optim_wrapper = dict(  # Optimizer wrapper config
    type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.02,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001),  # Weight decay of SGD
    clip_grad=None,  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

# param_scheduler is a field that configures methods of adjusting optimization hyperparameters 
param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # Use linear policy to warmup learning rate
        start_factor=0.001, # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=500),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type='MultiStepLR',  # Use multi-step learning rate policy during training
        by_epoch=True,  # The learning rate is updated by epoch
        begin=0,   # Start from the first epoch
        end=12,  # End at the 12th epoch
        milestones=[8, 11],  # Epochs to decay the learning rate
        gamma=0.1)  # The learning rate decay ratio
]

# Hook config
# Users can attach Hooks to training, validation, and testing loops to insert some operations during running.
# There are two different hook fields, one is default_hooks and the other is custom_hooks.
# default_hooks is a dict of hook configs, and they are the hooks must be required at the runtime. They have default priority which should not be modified. If not set, runner will use the default values. To disable a default hook, users can set its config to None.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=50),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
    visualization=dict(type='DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results

# custom_hooks = []를 사용하여 custom hook을 만들 수 있음

# Runtime config
default_scope = 'mmdet'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark, benchmark mode is good whenever your input sizes for your network do not vary
    mp_cfg=dict(  # Multi-processing config
        mp_start_method='fork',  # Use fork to start multi-processing threads. 'fork' usually faster than 'spawn' but maybe unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # Disable opencv multi-threads to avoid system being overloaded
    dist_cfg=dict(backend='nccl'),  # Distribution configs
)

vis_backends = [dict(type='LocalVisBackend')]  # Visualization backends. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type='LogProcessor',  # Log processor to process runtime logs
    window_size=50,  # Smooth interval of log values
    by_epoch=True)  # Whether to format logs with epoch type. Should be consistent with the train loop's type.

log_level = 'INFO'  # The level of logging.
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.

# ------------------Iter-based config----------------------

# To use iter-based training, users should modify the train_cfg, param_scheduler, train_dataloader, default_hooks, and log_processor.
# Iter-based training config
train_cfg = dict(
    _delete_=True,  # Ignore the base config setting (optional)
    type='IterBasedTrainLoop',  # Use iter-based training loop
    max_iters=90000,  # Maximum iterations
    val_interval=10000)  # Validation interval


# # Change the scheduler to iter-based
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90000,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

# Switch to InfiniteSampler to avoid dataloader restart
train_dataloader = dict(sampler=dict(type='InfiniteSampler'))

# Change the checkpoint saving interval to iter-based
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))

# Change the log format to iter-based
log_processor = dict(by_epoch=False)





# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'C:/Users/PJH/Desktop/AISM_mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# python tools/train.py plastic/my_custom_config.py
# python tools/train.py plastic/cascade-mask-rcnn_r50_fpn.py