# model settings
input_size = 300
num_classes = 80
model = dict(
    type='SingleStageDetector',
    pretrained='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    backbone=dict(
        type='MobileNetV2SSDFeatureExtractor',
        input_size=input_size,
        input_channel=32,
        last_channel=1280,
        width_mult=1.0,
        round_nearest=8,
        min_width=16,
        out_feature_indices=(19, None),
        out_feature_widths=(None, 128),
        activation_function='relu',
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHeadTFODAPI',
        in_channels=None,  # Provided by the backbone
        num_classes=num_classes,
        anchor_generator=dict(
            type='ODApiSSDAnchorGenerator',
            input_size=input_size,
            num_layers=2,
            min_scale=0.5,
            max_scale=0.95,
            aspect_ratios=[1.0],
        ),
        bbox_coder=dict(
            type='FasterRcnnBoxCoder',
            y_scale=10.0,
            x_scale=10.0,
            height_scale=5.0,
            width_scale=5.0,
        ),
      )
)
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
    nms=dict(type='nms', iou_threshold=0.6),
    min_bbox_size=0,
    score_thr=1e-8,
    max_per_img=10)
