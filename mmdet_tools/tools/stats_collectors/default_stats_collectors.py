from pathlib import Path
from typing import Union

from enot.utils.stats_collector import LogStatsCollector
from enot.utils.stats_collector import TensorboardStatsCollector


class TensorboardCOCOStatsCollector(TensorboardStatsCollector):
    def __init__(
            self,
            log_dir: Union[str, Path],
            train_log_frequency: int = None,
    ):
        super().__init__(log_dir, train_log_frequency)

    @property
    def logger(self):
        return self._logger


def get_stat_collectors(
        exp_dir: Union[str, Path],
        logger,
        postfix: str = '',
):

    stat_collectors = [
        LogStatsCollector(logger),
        TensorboardCOCOStatsCollector(exp_dir / f'tensorboard_{postfix}'),
    ]

    return stat_collectors
