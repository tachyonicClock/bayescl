import re
import sys
from pathlib import Path
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


def _short_metric_name(name: str, *, drop_granularity: bool) -> str:
    name = _METRIC_NAME_NOISE.sub("", name)
    if drop_granularity:
        name = _METRIC_GRANULARITY_SUFFIX.sub("", name)
    return name


_LOCATION_WIDTH = 36


def _format_record(record) -> str:
    # ``eval_tag`` is set via ``logger.contextualize`` around auxiliary eval
    # passes (early-stopping probes, OOD/shift eval) so they're visually
    # distinguishable from the real per-checkpoint eval, which carries no tag.
    tag = record["extra"].get("eval_tag")
    prefix = f"<yellow>[{tag}]</yellow> " if tag else ""
    # ``module`` (e.g. "agent_logger") rather than ``name`` (the fully
    # qualified "bayescl.metrics.agent_logger") -- shorter, and padded to a
    # fixed width so messages line up in a consistent column. If it's still
    # too long, truncate from the front (the line number and function name
    # are more useful than the module name) and mark the cut with ".." so it
    # doesn't read as a real, oddly-truncated module name.
    location = f"{record['module']}:{record['function']}:{record['line']}"
    if len(location) > _LOCATION_WIDTH:
        location = ".." + location[-(_LOCATION_WIDTH - 2) :].lstrip(":")
    location = location.replace("<", r"\<")  # "<" starts a loguru color tag
    # Colors are auto-disabled by loguru when the sink isn't a tty (e.g.
    # redirected to a file or piped to an agent), so this stays plain text
    # there without any extra handling on our part.
    return (
        "<green>{time:HH:mm:ss}</green>  "
        f"<cyan>{location:<{_LOCATION_WIDTH}}</cyan>  "
        f"{prefix}<level>{{message}}</level>\n"
    )


# Configured at import time (rather than lazily in ``AgentLogger.__init__``)
# so every ``logger.info`` call in the process uses this format -- including
# the ones that fire before an ``Experiment`` (and its ``AgentLogger``) is
# ever constructed.
logger.remove()
logger.add(sys.stderr, format=_format_record)

_file_sink_id: int | None = None


def set_log_file(path: Path) -> None:
    """(Re)point the file sink at ``path``, replacing any previous one.

    ``tune``/``test`` construct one ``Experiment`` per trial/seed, each with
    its own ``run_dir`` -- call this from there so every run gets its own log
    file instead of all of them piling into a single one.
    """
    global _file_sink_id
    if _file_sink_id is not None:
        logger.remove(_file_sink_id)
    _file_sink_id = logger.add(path, format=_format_record)


class AgentLogger(BaseLogger, SupervisedPlugin):
    """Plain, non-interactive logger for LLM/agent consumption.

    Prints one line per epoch or evaluation experience via loguru, instead
    of the tqdm progress bars used by ``InteractiveLogger``.
    """

    def __init__(self) -> None:
        super().__init__()
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
        # depth=1 attributes the log line to ``_flush``'s caller (e.g.
        # ``after_eval_exp``), which is what's actually meaningful here --
        # every flush otherwise reports this same helper as its origin.
        log = logger.opt(depth=1)
        if not self.metric_vals:
            log.info(header)
            return
        # Drop the granularity suffix (e.g. "_Epoch") only when it's safe to --
        # some metrics (e.g. loss_metrics(minibatch=True, epoch=True)) report
        # the same base name at multiple granularities in the same flush, and
        # collapsing those would silently clobber one value with the other.
        collapsed_counts: Dict[str, int] = {}
        for k in self.metric_vals:
            collapsed = _short_metric_name(k, drop_granularity=True)
            collapsed_counts[collapsed] = collapsed_counts.get(collapsed, 0) + 1

        parts = []
        for k, v in sorted(self.metric_vals.items()):
            collapsed = _short_metric_name(k, drop_granularity=True)
            name = collapsed if collapsed_counts[collapsed] == 1 else _short_metric_name(
                k, drop_granularity=False
            )
            parts.append(f"{name}={self._format_value(v)}")

        log.info(f"{header} | {', '.join(parts)}")
        self.metric_vals = {}

    def _exp_header(self, strategy, action: str) -> str:
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)
        if task_id is None:
            return f"{action} exp={exp_id} stream={stream}"
        return f"{action} exp={exp_id} task={task_id} stream={stream}"

    def before_training_exp(self, strategy, metric_values, **kwargs) -> None:
        super().before_training_exp(strategy, metric_values, **kwargs)
        logger.info(self._exp_header(strategy, "train start"))

    def after_training_epoch(self, strategy, metric_values, **kwargs) -> None:
        super().after_training_epoch(strategy, metric_values, **kwargs)
        exp_id = strategy.experience.current_experience
        epoch = strategy.clock.train_exp_epochs
        epoch_time = next(
            (
                v
                for k, v in self.metric_vals.items()
                if _short_metric_name(k, drop_granularity=True) == "Time"
            ),
            None,
        )
        if epoch_time:
            self.metric_vals["ips"] = len(strategy.experience.dataset) / epoch_time
        self._flush(f"train exp={exp_id} epoch={epoch}")

    def after_eval_exp(self, strategy, metric_values, **kwargs) -> None:
        super().after_eval_exp(strategy, metric_values, **kwargs)
        self._flush(self._exp_header(strategy, "eval"))

    def after_eval(self, strategy, metric_values, **kwargs) -> None:
        super().after_eval(strategy, metric_values, **kwargs)
        self._flush("eval stream summary")
