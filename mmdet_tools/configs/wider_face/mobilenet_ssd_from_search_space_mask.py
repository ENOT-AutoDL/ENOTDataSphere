_base_ = [
    '../_base_/models/ssd_mobilenet_from_search_space_mask.py', '../_base_/datasets/mask_face.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model settings
input_size = 300
num_classes = 2
dataset_type = 'CocoDataset'
data_root = 'data/masks/'
classes = ('face', 'mask')

model = dict(
    bbox_head=dict(num_classes=num_classes)
)

#lr_config = dict(
#    _delete_=True, policy='fixed', warmup='linear', warmup_iters=500, warmup_ratio=0.001)
# dataset settings
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
         type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18,
     ),
     dict(
         type='MinIoURandomCrop',
         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
         min_crop_size=0.3,
     ),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train_synth_real_augmented_filtred_08.json',
            img_prefix=data_root,
            classes=classes,
            pipeline=train_pipeline,
        )
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

optimizer = dict(_delete_=True, type='Adam', lr=1.0e-5, weight_decay=5e-4)
checkpoint_config = dict(interval=1)
total_epochs = 6

# evaluation (Copied just to keep important settings in one place in a high-level config)
evaluation = dict(interval=1, metric='bbox')
