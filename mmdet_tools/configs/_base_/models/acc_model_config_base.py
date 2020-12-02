# Search space config
# Build ss_config using build_searched_arch_cfg.py

# This config depends on founded architecture!!!

SEARCH_NET_CONFIG = {
    'blocks_count': (2, 3, 4, 3, 3, 1),
    'output_channels': (24, 32, 64, 96, 160, 320),
    'strides': (2, 2, 2, 1, 2, 1),
}

best_arch_ind = [1, 2, 0, 0, 1, 0, 4, 1, 5, 1, 0, 4, 0, 1, 4, 4]
BASIC_SEARCH_SPACE = (
    'MIB_k=3_t=6',
    'MIB_k=5_t=6',
    'MIB_k=7_t=6',
    'MIB_k=3_t=3',
    'MIB_k=5_t=3',
    'MIB_k=7_t=3',
    'conv1x1-skip',
)

BEST_ARCH = [BASIC_SEARCH_SPACE[i] for i in best_arch_ind]


input_size = 300
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(
        type='SimpleMobileNet',
        search_blocks=BEST_ARCH,
        mob_net_params=SEARCH_NET_CONFIG,
        out_feature_indices=(12, 16, None, None, None, None),
        out_feature_widths=(None, None, 512, 256, 256, 64),
        activation='relu',
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=(96, 320, 512, 256, 256, 64),
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
            target_stds=[0.1, 0.1, 0.2, 0.2])
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
    nms=dict(type='nms', iou_threshold=0.8),
    min_bbox_size=0,
    score_thr=1e-8,
    max_per_img=200
)
