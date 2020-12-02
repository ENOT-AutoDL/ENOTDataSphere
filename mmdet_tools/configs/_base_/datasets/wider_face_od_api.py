dataset_type = 'CocoDataset'
data_root = '/home/jupyter/work/resources/mmdetection_expasoft/data/widerface/'
classes = ('face',)
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'new_annotation_train_coco_small.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'new_annotation_val_coco_small.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'new_annotation_val_coco_small.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric='bbox')
