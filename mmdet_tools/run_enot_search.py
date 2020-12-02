import argparse
import os
import shutil
from argparse import Namespace
from pathlib import Path

from mmcv import Config
from enot_core.core.optimizers import Adam

from mmdet.enot_tools.enot_search import enot_search

parser = argparse.ArgumentParser()
parser.add_argument(
    "--rank",
    help="Process rank",
    type=int,
    default=0,
)
parser.add_argument(
    "--local_rank",
    help="Process rank",
    type=int,
    default=0,
)

search_cfg = {
  'epochs': 2,
  'dataloader_jobs': 4,
  'mmdet_config_path': './mmdet_tools/configs/wider_face/search_space_ssd_masks_search.py',
  'experiment_dir': './ssd_search_space_pretrain_adam/',
  'pretrain_checkpoint_path':'/home/jupyter/work/resources/models/pretrain_3_masks.pth' #'./mmdetection_expasoft/ssd_search_space_pretrain_adam/experiments/conv_pretrain/checkpoint-3.pth'
}
experiment_cfg = {
  'dir': './mmdetection_expasoft/experiments/conv_search',
}


def make_pretrain_args(tmp_dir, pretrain_cfg_, experiment_cfg_):
    return Namespace(
        exp_dir=tmp_dir/experiment_cfg_['dir'],
        jobs=pretrain_cfg_['dataloader_jobs'],
        epochs=pretrain_cfg_['epochs'],
        rank=0,
        local_rank=0,
        seed=0,
    )


def run_enot_search(latency_loss_weight=0.0):
    exp_dir = search_cfg['experiment_dir']
    mmdet_config = search_cfg['mmdet_config_path']
    pretrain_checkpoint = search_cfg['pretrain_checkpoint_path']
    # Copy runner
    os.makedirs(f'{exp_dir}{experiment_cfg["dir"]}', exist_ok=True)
    file_name = Path(__file__).name
    shutil.copy(__file__, f'{exp_dir}{experiment_cfg["dir"]}/{os.getpid()}_{file_name}')
    # Copy mmdet cfg file.
    cfg = Config.fromfile(mmdet_config)
    cfg.dump(f'{exp_dir}{experiment_cfg["dir"]}/{os.getpid()}_search_mmdetcfg.py')

    args = make_pretrain_args(
        Path(exp_dir),
        search_cfg,
        experiment_cfg,
    )
    enot_search(
        model_config_path=mmdet_config,
        checkpoint_path=pretrain_checkpoint,
        experiment_args=args,
        optimizer_class=Adam,
        opt_params={
            'lr': 0.0008
        },
        scheduler=None,
        scheduler_params=None,
        batch_size=32,
        latency_loss_weight=latency_loss_weight,
    )
