# https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/

from contextlib import nullcontext

from avalanche.training.templates.update_type import SGDUpdate
from loguru import logger
from torch.utils.flop_counter import FlopCounterMode


def training_epoch(self, **kwargs):
    """Training epoch.

    :param kwargs:
    :return:
    """
    for i, self.mbatch in enumerate(self.dataloader):
        if self._stop_training:
            break
        if i == 0:
            flop_counter_forward = FlopCounterMode(display=False, depth=None)
            flop_counter_backward = FlopCounterMode(display=False, depth=None)
        else:
            flop_counter_forward = nullcontext()
            flop_counter_backward = nullcontext()

        self._unpack_minibatch()
        self._before_training_iteration(**kwargs)

        self.optimizer.zero_grad()
        self.loss = self._make_empty_loss()

        # Forward
        with flop_counter_forward:
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

        # Loss & Backward
        with flop_counter_backward:
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

        # Optimization step
        self._before_update(**kwargs)
        self.optimizer_step()
        self._after_update(**kwargs)

        self._after_training_iteration(**kwargs)

        if i == 0:
            n = len(self.mbatch[0])
            gflop_forward = flop_counter_forward.get_total_flops() / n / 2 / 1e9
            gflop_backward = flop_counter_backward.get_total_flops() / n / 2 / 1e9
            logger.info(f"GFLOP (forward) : {gflop_forward:.2f}")
            logger.info(f"GFLOP (backward): {gflop_backward:.2f}")


def monkey_patch_count_flops():
    SGDUpdate.training_epoch = training_epoch
