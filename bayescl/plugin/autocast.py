from typing import Any

import torch
from avalanche.training.plugins import SupervisedPlugin


class Autocast(SupervisedPlugin):
    """Wraps each forward pass and loss computation -- train and eval -- in
    ``torch.autocast``, leaving backward and the optimizer step at full
    precision. This is the standard mixed-precision recipe: the ops PyTorch
    considers numerically unsafe (e.g. softmax) are auto-upcast within the
    context regardless, and bf16 (unlike fp16) has fp32's exponent range so
    no ``GradScaler`` is needed.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype
        self._ctx: torch.autocast | None = None

    def _enter(self, strategy: Any) -> None:
        self._ctx = torch.autocast(device_type=strategy.device.type, dtype=self.dtype)
        self._ctx.__enter__()

    def _exit(self) -> None:
        self._ctx.__exit__(None, None, None)
        self._ctx = None

    def before_forward(self, strategy: Any, *args, **kwargs) -> None:
        self._enter(strategy)

    def after_forward(self, strategy: Any, *args, **kwargs) -> None:
        # Cast back to float32 immediately so every downstream consumer
        # (masking, loss, captured-logits metrics/numpy conversions) sees the
        # same dtype it always has -- only the forward matmuls run in bf16.
        strategy.mb_output = strategy.mb_output.float()

    def before_backward(self, strategy: Any, *args, **kwargs) -> None:
        self._exit()

    def before_eval_forward(self, strategy: Any, *args, **kwargs) -> None:
        self._enter(strategy)

    def after_eval_forward(self, strategy: Any, *args, **kwargs) -> None:
        strategy.mb_output = strategy.mb_output.float()

    def after_eval_iteration(self, strategy: Any, *args, **kwargs) -> None:
        self._exit()
