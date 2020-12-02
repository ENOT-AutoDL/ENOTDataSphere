import argparse
import os
import shutil
from argparse import Namespace
from pathlib import Path

from mmcv import Config
from enot_core.core.optimizers import Adam

from mmdet.enot_tools.enot_pretrain import enot_pretrain


parser = argparse.ArgumentParser()
parser.add_argument(
    '--rank',
    help='Process rank',
    type=int,
    default=0,
)
parser.add_argument(
    '--local_rank',
    help='Process rank',
    type=int,
    default=0,
)

pretrain_cfg = {
  'epochs': 3,
  'dataloader_jobs': 4,
  'mmdet_config_path': '/home/jupyter/work/resources/mmdet_tools/configs/wider_face/search_space_ssd_masks.py',
  'experiment_dir': '/home/jupyter/work/resources/ssd_search_space_pretrain_adam/',
}

experiment_cfg = {
  'dir': 'experiments/conv_pretrain',
}


def make_pretrain_args(tmp_dir, pretrain_cfg_, experiment_cfg_):
    return Namespace(
        exp_dir=tmp_dir/experiment_cfg_['dir'],
        jobs=pretrain_cfg_['dataloader_jobs'],
        epochs=pretrain_cfg_['epochs'],
        rank=0,
        local_rank=0,
        seed=0,
        checkpoint_path='/home/jupyter/work/resources/pretrain_mask.pth',
    )


def run_enot_pretrain():
    exp_dir = pretrain_cfg['experiment_dir']
    mmdet_config = pretrain_cfg['mmdet_config_path']
    # Copy runner
    os.makedirs(f'{exp_dir}{experiment_cfg["dir"]}', exist_ok=True)
    file_name = Path(__file__).name
    shutil.copy(__file__, f'{exp_dir}{experiment_cfg["dir"]}/{os.getpid()}_{file_name}')
    # Copy mmdet cfg file.
    cfg = Config.fromfile(mmdet_config)
    cfg.dump(f'{exp_dir}{experiment_cfg["dir"]}/{os.getpid()}_mmdetcfg.py')

    args = make_pretrain_args(
        Path(exp_dir),
        pretrain_cfg,
        experiment_cfg,
    )

    enot_pretrain(
        model_config_path=mmdet_config,
        experiment_args=args,
        optimizer_class=Adam,
        opt_params={
            'lr': 8e-4,
            'weight_decay': 5e-4,
        },
        scheduler=None,
        scheduler_params=None,
        batch_size=32,
    )
