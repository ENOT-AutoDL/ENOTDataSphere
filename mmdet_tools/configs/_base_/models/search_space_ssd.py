# Search space config
# Build ss_config using provided code.

# from enot.construction_utils import make_ss_config
# from enot.utils.config_helper import SEARCH_NET_CONFIG
# from enot.utils.config_helper import BASIC_SEARCH_SPACE
# cfg = make_ss_config(SEARCH_NET_CONFIG, BASIC_SEARCH_SPACE, (128, 128, 16))

# We cannot generate this config in search_space_ssd.py because of mmdetection config workflow.
# mmdetection cannot merge this config if there are other imports.
# So we have to generate ss_config manually and then copy in this mmdet config.


SEARCH_NET_CONFIG = {
    'blocks_count': (4, 4, 4, 4, 4, 1),
    'output_channels': (24, 32, 64, 96, 160, 320),
    'strides': (2, 2, 2, 1, 2, 1),
}

BASIC_SEARCH_SPACE = (
    'MIB_k=3_t=6',
    'MIB_k=5_t=6',
    'MIB_k=7_t=6',
    'MIB_k=3_t=3',
    'MIB_k=5_t=3',
    'MIB_k=7_t=3',
    'conv1x1-skip',
)


input_size = 300
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(
        type='MobileNetSearchSpace',
        search_blocks=BASIC_SEARCH_SPACE,
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
