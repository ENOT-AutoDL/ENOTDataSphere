import os
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.distributed as dist
from enot import phases
from enot.phases.phases import _create_permutation_sampler
from pytorch_utils import prepare_log
from pytorch_utils.dataloaders import CudaDataLoader
from pytorch_utils.distributed_utils import synchronize_model_with_checkpoint
from pytorch_utils.distributed_utils import torch_save
from pytorch_utils.schedulers import Scheduler
from pytorch_utils.train_utils import init_exp_dir
from mmcv import Config
from torch.utils.data import DistributedSampler

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


def enot_pretrain(
        model_config_path: str,
        experiment_args: Namespace,
        optimizer_class,
        opt_params: Dict[str, Any],
        scheduler: Optional[Scheduler],
        scheduler_params: Optional[Dict[str, Any]],
        batch_size: int,
) -> None:
    """
    Using mmdetection config build model and dataset and start pretrain phase
    """
    # Build SearchSpace using mmdetection config and builder

    # Initialize enot runner params, multigpu params and etc.
    exp_dir = init_exp_dir(experiment_args)
    logger = prepare_log(log_path=exp_dir / "log_pretrain.txt")

    logger.info("Initial preparation ready")
    # For multigpu we use distributed data sampler.
    if not dist.is_initialized():
        distributed_sampler = False
    else:
        distributed_sampler = True

    # Load config and build model
    cfg = Config.fromfile(model_config_path)
    # Current model is SearchSpace object with SearchSpace backbone.
    search_space = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if 'checkpoint_path' in experiment_args:
        print(experiment_args.checkpoint_path)
        search_space.load_state_dict(
            torch.load(
                experiment_args.checkpoint_path,
                map_location="cpu",
            )["model"],
        )
    # User has to manually place model to cuda.
    search_space.cuda()
    # synchronize models weights for multigpu
    synchronize_model_with_checkpoint(search_space)
    logger.info("Model ready")

    # TODO multiple dataset problem
    # Now supported only single dataset configs.
    train_datasets = [build_dataset(cfg.data.train)]
    valid_dataset = [build_dataset(cfg.data.test)]

    # By default mmdetection dataset use as output inputs dict, enot framework expect pair (inputs, labels).
    # For training dataloader outputs is (train_dict, None) pair.
    # For validation dataloader output is (validation_dict, image_id), we use these ids for sorting model results
    # these sorted results will be evaluated using evaluate method of dataset.
    # We must sort these outputs, because after multigpu validation results are shuffled.
    sampler_valid = DistributedSampler(valid_dataset[0], shuffle=False) if distributed_sampler else None
    valid_dataloader = CudaDataLoader(
        MMDetDatasetEnumerateWrapper(valid_dataset[0]),
        batch_size=1,
        collate_fn=valid_nas_collate_function,
        sampler=sampler_valid,
    )
    sampler_train = DistributedSampler(train_datasets[0], shuffle=False) if distributed_sampler else None
    train_dataloader = CudaDataLoader(
        train_datasets[0],
        batch_size=batch_size,
        collate_fn=nas_collate_function,
        num_workers=experiment_args.jobs,
        sampler=sampler_train,
    )

    # User must manually place input tensors to cuda.
    logger.info('Dataloaders ready')

    # For pretrain phase we update SearchSpace params.
    optimizer = optimizer_class(params=search_space.model_parameters(), **opt_params)
    if scheduler:
        scheduler = scheduler(optimizer, **scheduler_params)

    logger.info('Train schedule ready')

    torch_save(
        {
            'epoch': 0,
            'model': search_space.state_dict(),
        },
        os.path.join(exp_dir, 'checkpoint-0.pth'),
    )

    # dummy accuracy function. Because we cannot evaluate COCO metrics on the fly.
    def accuracy(pred_labels, labels):
        return 0.0

    stats_collectors = get_stat_collectors(
        exp_dir=Path(exp_dir),
        logger=logger,
        postfix='pretrain',
    )
    coco_eval = COCOEval(
        model=search_space,
        valid_loader=valid_dataloader,
        metric_function=partial(custom_coco_evaluation, dataset=valid_dataset[0]),
        validation_forward_wrapper=custom_valid_forward_logic,
        logger=logger,
        phase='pretrain',
        sampler=_create_permutation_sampler(search_space.search_blocks, True),
        tensorboard_collector=stats_collectors[-1],
    )
    stats_collectors.append(coco_eval)

    phases.pretrain(
        search_space=search_space,
        exp_dir=exp_dir,
        train_loader=train_dataloader,
        valid_loader=[],
        optimizer=optimizer,
        scheduler=scheduler,
        stats_collectors=stats_collectors,
        loss_function=parse_losses,
        epochs=experiment_args.epochs,
        metric_function=accuracy,
        validation_forward_wrapper=custom_valid_forward_logic,
        train_forward_wrapper=custom_train_forward_logic,
    )
