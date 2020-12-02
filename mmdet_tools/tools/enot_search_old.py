from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import torch
from enot import phases
from pytorch_utils import prepare_log
from pytorch_utils.dataloaders import CudaDataLoader
from pytorch_utils.distributed_utils import synchronize_model_with_checkpoint
from pytorch_utils.schedulers import Scheduler
from pytorch_utils.train_utils import init_exp_dir
from mmcv import Config

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from .nas_tools import MMDetDatasetEnumerateWrapper
from .nas_tools import custom_coco_evaluation
from .nas_tools import custom_train_forward_logic
from .nas_tools import custom_valid_forward_logic
from .nas_tools import nas_collate_function
from .nas_tools import parse_losses
from .nas_tools import valid_nas_collate_function
from .stats_collectors.default_stats_collectors import get_stat_collectors
from .stats_collectors.enot_coco_eval import COCOEval


def enot_search(
        model_config_path: str,
        experiment_args: Namespace,
        checkpoint_path: str,
        batch_size: int,
        optimizer_class,
        opt_params: Dict[str, Any],
        scheduler: Optional[Scheduler],
        scheduler_params: Optional[Dict[str, Any]],
        latency_loss_weight: float,
):
    """using mmdetection config build model and dataset and start search phase.
    Note that for search phase we should use different mmdet config where validation and train set are the same.
    (mmdetection build train and val set in different way, but we must use val data for search)
    """
    # Initialize enot runner params, multigpu params and etc.
    exp_dir = init_exp_dir(experiment_args)
    logger = prepare_log(log_path=exp_dir / 'log_pretrain.txt')
    logger.info('Initial preparation ready')

    # build Search Space and load weights from pretrain phase
    # Current model is SearchSpace object with SearchSpace backbone.
    cfg = Config.fromfile(model_config_path)
    search_space = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    search_space.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location='cpu',
        )['model'],
        strict=True,
    )
    search_space.cuda()
    synchronize_model_with_checkpoint(search_space)
    logger.info('Model ready')

    # In Search Phase we do not use multigpu.
    # By default mmdetection dataset use as output inputs dict, enot framework expect pair (inputs, labels).
    # For training dataloader outputs is (train_dict, None) pair.
    # For validation dataloader output is (validation_dict, image_id), we use these ids for sorting model results
    # these sorted results will be evaluated using evaluate method of dataset.
    # We must sort these outputs, because after multigpu validation results are shuffled.
    valid_dataset = [build_dataset(cfg.data.test)]
    train_dataset = [build_dataset(cfg.data.train)]
    valid_dataloader = CudaDataLoader(
        MMDetDatasetEnumerateWrapper(valid_dataset[0]),
        batch_size=1,
        collate_fn=valid_nas_collate_function,
    )
    train_dataloader = CudaDataLoader(
        train_dataset[0],
        batch_size=batch_size,
        collate_fn=nas_collate_function,
    )
    zero = [[0,],]*len(search_space.backbone.search_variants_containers)
    search_space.sample(zero, None)
    search_space.initialize_latency(next(iter(train_dataloader))[0])
    print("MAC's: ", search_space.forward_latency.item())

    # User must manually place input tensors to cuda.

    logger.info('Dataloaders ready')

    # For search phase we update SearchSpace architecture params.
    optimizer = optimizer_class(search_space.architecture_parameters(), **opt_params)
    if scheduler:
        scheduler = scheduler(optimizer, **scheduler_params)
    logger.info('Train schedule ready')

    # dummy accuracy function. Because we cannot evaluate COCO metrics on the fly.
    def accuracy(pred_labels, labels):
        return 0.0

    stats_collectors = get_stat_collectors(
        exp_dir=Path(exp_dir),
        logger=logger,
        postfix='search',
    )
    coco_eval = COCOEval(
        model=search_space,
        valid_loader=valid_dataloader,
        metric_function=partial(custom_coco_evaluation, dataset=valid_dataset[0]),
        validation_forward_wrapper=custom_valid_forward_logic,
        logger=logger,
        phase='search',
        tensorboard_collector=stats_collectors[-1],
    )
    stats_collectors.append(coco_eval)

    phases.search(
        search_space=search_space,
        exp_dir=exp_dir,
        search_loader=train_dataloader,
        optimizer=optimizer,
        loss_function=parse_losses,
        epochs=experiment_args.epochs,
        stats_collectors=stats_collectors,
        metric_function=accuracy,
        latency_loss_weight=latency_loss_weight,
        scheduler=scheduler,
        validation_forward_wrapper=custom_valid_forward_logic,
        train_forward_wrapper=custom_train_forward_logic,
        valid_loader=[],
    )
