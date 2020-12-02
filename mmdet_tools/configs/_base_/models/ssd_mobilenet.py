# model settings
input_size = 300
model = dict(
    type='SingleStageDetector',
    pretrained='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    backbone=dict(
        type='MobileNetV2SSDFeatureExtractor',
        input_size=300,
        input_channel=32,
        last_channel=1280,
        width_mult=1.0,
        round_nearest=8,
        min_width=16,
        out_feature_indices=(14, 19, None, None, None, None),
        out_feature_widths=(None, None, 512, 256, 256, 64),
        activation_function='relu',
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=(96, 1280, 512, 256, 256, 64),
        num_classes=80,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[16, 30, 60, 100, 150, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)

test_cfg = dict(
    nms=dict(type='nms', iou_threshold=0.8),
    min_bbox_size=0,
    score_thr=1e-8,
    max_per_img=200
)
