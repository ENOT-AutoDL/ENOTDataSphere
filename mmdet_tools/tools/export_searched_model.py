import torch
from enot.models.simple_block_model_builders import build_frozen_simple_block_model
from mmcv import Config

from mmdet.models import build_detector


def extract_model_cfg_from_search_space(search_space_model_cfg, search_phase_ckpt):
    """Extract model configuration for simple detector from search space"""
    cfg = Config.fromfile(search_space_model_cfg)
    search_net_config = cfg.SEARCH_NET_CONFIG
    op_names_in_ss = cfg.BASIC_SEARCH_SPACE

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.load_state_dict(
        torch.load(search_phase_ckpt)['model'],
        strict=True,
    )
    searched_arch_indices = model.best_architecture_int
    best_operation_names = [op_names_in_ss[index].replace('[s]', '') for index in searched_arch_indices]

    cfg = build_frozen_simple_block_model(
        in_channels=16,
        blocks_op_name=list(best_operation_names),
        blocks_out_channels=search_net_config['output_channels'],
        blocks_count=search_net_config['blocks_count'],
        blocks_stride=search_net_config['strides'],
    )

    return cfg


def extract_model_ckpt_from_search_space(search_space_model_cfg, model_cfg, search_phase_ckpt):
    """Extract model checkpoint for simple detector from search space"""
    cfg_ss = Config.fromfile(search_space_model_cfg)
    model = build_detector(cfg_ss.model, train_cfg=cfg_ss.train_cfg, test_cfg=cfg_ss.test_cfg)
    model.load_state_dict(
        torch.load(search_phase_ckpt)['model'],
        strict=True,
    )
    cfg_model = Config.fromfile(model_cfg)
    model_searched = build_detector(cfg_model.model, train_cfg=cfg_model.train_cfg, test_cfg=cfg_model.test_cfg)
    best_arch_ind = cfg_model.best_arch_ind if hasattr(cfg_model, 'best_arch_ind') else None
    best_arch_ind = best_arch_ind if best_arch_ind else model.best_architecture_int

    state_dict_searched = model.state_dict(
        architecture=best_arch_ind,
        reference_model=model_searched,
    )

    return state_dict_searched
