import re
import sys
from typing import Any, Dict

from avalanche.evaluation.metric_results import AlternativeValues, TensorImage
from avalanche.evaluation.metric_utils import phase_and_task, stream_type
from avalanche.logging import BaseLogger
from avalanche.training.plugins import SupervisedPlugin
from loguru import logger
from torch import Tensor

_SCALAR_TYPES = (int, float, str)

# Avalanche metric names are built as
# "<name>/<train|eval>_phase/<stream>_stream[/TaskNNN][/ExpNNN]" -- everything
# from "/<phase>_phase" onward duplicates what ``_exp_header`` already prints,
# and the granularity suffix (e.g. "_Exp") duplicates which flush call this is.
_METRIC_NAME_NOISE = re.compile(r"/(train|eval)_phase/\w+_stream(?:/Task\d+)?(?:/Exp\d+)?$")
_METRIC_GRANULARITY_SUFFIX = re.compile(r"_(Epoch|Exp|MB|Stream)$")


def _short_metric_name(name: str) -> str:
    name = _METRIC_NAME_NOISE.sub("", name)
    return _METRIC_GRANULARITY_SUFFIX.sub("", name)


_format_configured = False


def _configure_format() -> None:
    """Swap loguru's default sink for a leaner one, once.

    The default format prints ``module:function:line`` on every line, which
    is redundant noise for training logs that already state their own
    context (phase/experience/epoch) in the message body.
    """
    global _format_configured
    if _format_configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {message}",
    )
    _format_configured = True


class AgentLogger(BaseLogger, SupervisedPlugin):
    """Plain, non-interactive logger for LLM/agent consumption.

    Prints one line per epoch or evaluation experience via loguru, instead
    of the tqdm progress bars used by ``InteractiveLogger``.
    """

    def __init__(self) -> None:
        super().__init__()
        _configure_format()
        self.metric_vals: Dict[str, Any] = {}

    def log_single_metric(self, name: str, value: Any, x_plot: int) -> None:
        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(*_SCALAR_TYPES, Tensor)
        if value is None or isinstance(value, TensorImage):
            return
        if isinstance(value, Tensor) and value.numel() != 1:
            # Non-scalar tensors (e.g. confusion matrices) aren't useful as text.
            return
        self.metric_vals[name] = value

    def _format_value(self, value: Any) -> str:
        if isinstance(value, Tensor):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _flush(self, header: str) -> None:
        if self.metric_vals:
            metrics = ", ".join(
                f"{_short_metric_name(k)}={self._format_value(v)}"
                for k, v in sorted(self.metric_vals.items())
            )
            logger.info(f"{header} | {metrics}")
            self.metric_vals = {}
        else:
            logger.info(header)

    def _exp_header(self, strategy, action: str) -> str:
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)
        if task_id is None:
            return f"{action} exp={exp_id} stream={stream}"
        return f"{action} exp={exp_id} task={task_id} stream={stream}"

    def before_training(self, strategy, metric_values, **kwargs) -> None:
        super().before_training(strategy, metric_values, **kwargs)
        logger.info("-- training started --")

    def before_training_exp(self, strategy, metric_values, **kwargs) -> None:
        super().before_training_exp(strategy, metric_values, **kwargs)
        logger.info(self._exp_header(strategy, "train start"))

    def after_training_epoch(self, strategy, metric_values, **kwargs) -> None:
        super().after_training_epoch(strategy, metric_values, **kwargs)
        exp_id = strategy.experience.current_experience
        epoch = strategy.clock.train_exp_epochs
        self._flush(f"train exp={exp_id} epoch={epoch}")

    def after_training(self, strategy, metric_values, **kwargs) -> None:
        super().after_training(strategy, metric_values, **kwargs)
        logger.info("-- training ended --")

    def before_eval(self, strategy, metric_values, **kwargs) -> None:
        super().before_eval(strategy, metric_values, **kwargs)
        logger.info("-- eval started --")

    def before_eval_exp(self, strategy, metric_values, **kwargs) -> None:
        super().before_eval_exp(strategy, metric_values, **kwargs)

    def after_eval_exp(self, strategy, metric_values, **kwargs) -> None:
        super().after_eval_exp(strategy, metric_values, **kwargs)
        self._flush(self._exp_header(strategy, "eval"))

    def after_eval(self, strategy, metric_values, **kwargs) -> None:
        super().after_eval(strategy, metric_values, **kwargs)
        self._flush("-- eval ended --")
