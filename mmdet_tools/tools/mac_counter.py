from typing import List

from mmcv import Config
from enot_utils.dataloaders import CudaDataLoader
from enot_utils.distributed_utils import synchronize_model_with_checkpoint

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from .nas_tools import nas_collate_function


def get_mac(
        model_config_path: str,
        arch_int: List[int],
        batch_size: int = 32,
):
    """using mmdetection config build model and dataset and start search phase.
    Note that for search phase we should use different mmdet config where validation and train set are the same.
    (mmdetection build train and val set in different way, but we must use val data for search)
    """

    # build Search Space and load weights from pretrain phase
    # Current model is SearchSpace object with SearchSpace backbone.
    cfg = Config.fromfile(model_config_path)
    search_space = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    search_space.cuda()
    synchronize_model_with_checkpoint(search_space)

    # In Search Phase we do not use multigpu.
    # By default mmdetection dataset use as output inputs dict, enot framework expect pair (inputs, labels).
    # For training dataloader outputs is (train_dict, None) pair.
    # For validation dataloader output is (validation_dict, image_id), we use these ids for sorting model results
    # these sorted results will be evaluated using evaluate method of dataset.
    # We must sort these outputs, because after multigpu validation results are shuffled.
    train_dataset = [build_dataset(cfg.data.train)]

    train_dataloader = CudaDataLoader(
        train_dataset[0],
        batch_size=batch_size,
        collate_fn=nas_collate_function,
    )
    arch_int = [[x,] for x in arch_int]
    if len(arch_int) != len(search_space.backbone.search_variants_containers):
        raise RuntimeError("Length of passed archetecture is {len(arch_int)}, but in model {len(search_space.backbone.search_variants_containers)}")
    search_space.sample(arch_int, None)
    search_space.initialize_latency(next(iter(train_dataloader))[0])
    macs = search_space.forward_latency.item()
    print("MAC's millions: ", macs)
    return macs
