# Moneky patch `BaseSGDTemplate` to fix error in the training loop when `eval_streams`
# is not None.
from typing import Any

from avalanche.training.templates import BaseSGDTemplate


def _patched_train_exp(self, experience: Any, eval_streams=None, **kwargs):
    """Training loop over a single Experience object.

    :param experience: CL experience information.
    :param eval_streams: list of streams for evaluation.
        If None: use the training experience for evaluation.
        Use [] if you do not want to evaluate during training.
    :param kwargs: custom arguments.
    """
    if eval_streams is None:
        eval_streams = [experience]

    for _ in range(self.train_epochs):
        self._before_training_epoch(**kwargs)

        if self._stop_training:  # Early stopping
            self._stop_training = False
            break

        self.training_epoch(**kwargs)
        self._after_training_epoch(**kwargs)


BaseSGDTemplate._train_exp = _patched_train_exp
