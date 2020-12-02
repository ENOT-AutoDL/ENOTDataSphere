from logging import Logger
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import torch
import torch.distributed as dist
from enot.phases.utils import ForwardWrapper
from enot.utils.stats_collector.stats_collector import StatsCollector
from enot_core.core.search_space import SearchSpace
from enot_core.pretrain_utils import PermutationSampler
from pytorch_utils.distributed_utils import MetricsTracker
from pytorch_utils.distributed_utils import ResultHandler
from pytorch_utils.distributed_utils import is_master
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from .default_stats_collectors import TensorboardCOCOStatsCollector


def _search_space_pretrain_validation_step(
        search_space: SearchSpace,
        sampler: PermutationSampler,
        inputs: torch.Tensor,
        labels: Any,
        forward_wrapper: Callable[[torch.nn.Module, Any], Any],
):
    r"""Method for custom validation. Sample architectures and gather results in dictionary.
    """
    architecture_results = []
    with torch.no_grad():
        for i, indices in enumerate(sampler):
            search_space.sample(forward_indices=indices)
            output = forward_wrapper(search_space, inputs)
            architecture_results.append((output, labels))
    return architecture_results


def _search_space_search_validation_step(
        search_space: SearchSpace,
        sampler: PermutationSampler,
        inputs: torch.Tensor,
        labels: Any,
        forward_wrapper: Callable[[torch.nn.Module, Any], Any],
):
    r"""Gather architecture outputs.
        List used for compatibility with train evaluation where used multiple architectures
    """
    forward_indices = [[x] for x in search_space.best_architecture_int]
    search_space.eval(forward_indices=forward_indices)
    with torch.no_grad():
        output = forward_wrapper(search_space, inputs)
        architecture_results = [(output, labels)]

    return architecture_results


def _delete_duplicates(results: List[List[Any]]) -> List[List[Any]]:
    for arch_result in results:
        duplicated_indices = []
        duplicated_predictions = {}
        for idx, prediction in enumerate(arch_result):
            prediction_id = prediction[-1][0]
            if prediction_id in duplicated_predictions:
                duplicated_indices.append(idx)
            else:
                duplicated_predictions[prediction_id] = 1
        for ind in sorted(duplicated_indices, reverse=True):
            arch_result.pop(ind)
    return results


class COCOEval(StatsCollector):
    """
    Class for evaluation COCO-like datasets
    """
    def __init__(
            self,
            model: SearchSpace,
            valid_loader: DataLoader,
            metric_function: Callable[[List[Any]], Dict[str, float]],
            validation_forward_wrapper: ForwardWrapper,
            logger: Logger,
            phase: str,
            sampler: PermutationSampler = None,
            tensorboard_collector: TensorboardCOCOStatsCollector = None,
    ):
        StatsCollector.__init__(self)
        self._metric_function = metric_function
        self._model = model.eval()
        self._valid_loader = valid_loader
        self._sampler = sampler
        self._validation_forward_wrapper = validation_forward_wrapper
        self._logger = logger
        self._result_handler = ResultHandler()
        self._get_validation_func(phase)
        self.tensorboard_collector = tensorboard_collector

    def on_epoch_end(self, epoch: int, stats: Dict = None) -> None:
        self._calculate_metrics(epoch, stats)

    def _calculate_metrics(self, epoch: int, stats: Dict = None) -> None:
        valid_metrics_tracker = MetricsTracker()
        valid_metrics_tracker.reset()
        total_arch_results = self._eval()

        self._result_handler.write_result(total_arch_results)
        if dist.is_initialized():
            dist.barrier()
        self._result_handler.collect_results()
        if is_master():
            _delete_duplicates(self._result_handler.merged_results)
            eval_res = self._metric_function(self._result_handler.merged_results)
            valid_size = len(self._valid_loader)
            valid_metrics_tracker.update(eval_res, valid_size)
            self._logging_metrics(valid_metrics_tracker.get_metrics(), epoch)

    def _logging_metrics(self, metrics: Dict[str, float], step):
        for map_name, map_value in metrics.items():
            self._logger.info(f'Test {map_name}: {map_value}')
            if self.tensorboard_collector:
                tag = f'MAPs/{map_name}'
                self.tensorboard_collector.logger.add_scalar(tag, map_value, step)

    def _get_validation_func(self, phase_name: str):
        if phase_name.lower() == 'pretrain':
            self._validation_step = _search_space_pretrain_validation_step
        elif phase_name.lower() == 'search':
            self._validation_step = _search_space_search_validation_step
        else:
            raise NotImplementedError(
                f'Evaluation implemented only for pretrain and search phase, but get {phase_name} phase')

    def _eval(self):
        total_arch_results = []
        for step_number, (inputs, labels) in enumerate(tqdm(self._valid_loader, disable=not is_master())):
            arch_results = self._validation_step(
                search_space=self._model,
                sampler=self._sampler,
                inputs=inputs,
                forward_wrapper=self._validation_forward_wrapper,
                labels=labels,
            )
            if total_arch_results:
                for arch_res, cur_arch_res in zip(total_arch_results, arch_results):
                    arch_res.append(cur_arch_res)
            else:
                total_arch_results = [[res] for res in arch_results]
        return total_arch_results
