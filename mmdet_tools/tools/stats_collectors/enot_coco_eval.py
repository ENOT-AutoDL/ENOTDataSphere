from logging import Logger
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from enot.utils.stats_collector.stats_collector import StatsCollector

from .default_stats_collectors import TensorboardCOCOStatsCollector


class COCOEval(StatsCollector):
    """
    Class for evaluation COCO-like datasets
    """
    def __init__(
            self,
            metric_function: Callable[[List[Any]], Dict[str, float]],
            logger: Logger,
            tensorboard_collector: TensorboardCOCOStatsCollector = None,
    ):
        super().__init__()

        self._validation_results = dict()

        self._metric_function = metric_function
        self._logger = logger
        self.tensorboard_collector = tensorboard_collector

    @property
    def need_validation_batch_result(self) -> bool:
        return True

    def on_epoch_start(self, epoch: int) -> None:
        self._validation_results = dict()

    def on_epoch_end(self, epoch: int, stats: Dict) -> None:
        validation_results = list(self._validation_results.values())
        current_metrics = self._metric_function(validation_results)
        self._logging_metrics(current_metrics, epoch)

    def on_validation_batch_result(
            self,
            batch: int,
            predicted: Any,
            original: Any,
            process_index: int,
            sample_index: Optional[int],
    ) -> None:
        arch_results = self._validation_results.setdefault(sample_index, [])
        arch_results.append((predicted, original))

    def _logging_metrics(self, metrics: Dict[str, float], step: int):
        for map_name, map_value in metrics.items():
            self._logger.info(f'Test {map_name}: {map_value}')
            if self.tensorboard_collector:
                tag = f'MAPs/{map_name}'
                self.tensorboard_collector.logger.add_scalar(tag, map_value, step)
