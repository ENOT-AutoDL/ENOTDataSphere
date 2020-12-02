from .search_space_ssd import BASIC_SEARCH_SPACE
from .search_space_ssd import SEARCH_NET_CONFIG
# Search space config
# Build ss_config using build_searched_arch_cfg.py

# This config depends on founded architecture!!!

BEST_ARCH = [0, 0, 3, 1, 2, 5, 4, 1, 2, 2, 0, 2, 5, 2, 5, 5, 5, 0, 2, 2, 3]
BEST_ARCH = [BASIC_SEARCH_SPACE[i] for i in BEST_ARCH]

input_size = 300
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(
        type='SimpleMobileNet',
        search_blocks=BEST_ARCH,
        mob_net_params=SEARCH_NET_CONFIG,
        out_feature_indices=(16, 21, None, None, None, None),
        out_feature_widths=(None, None, 512, 256, 256, 64),
        activation='relu',
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHeadTFODAPI',
        in_channels=(96, 320, 512, 256, 256, 64),
        num_classes=80,
        anchor_generator=dict(
            type='ODApiSSDAnchorGenerator',
            input_size=input_size,
            num_layers=6,
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
        )
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
    debug=False
)

test_cfg = dict(
    nms=dict(type='nms', iou_threshold=0.6),
    min_bbox_size=0,
    score_thr=1e-8,
    max_per_img=10
)
