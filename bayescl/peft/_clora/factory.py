import torch
from torch import Tensor, nn
from typeguard import typechecked

from bayescl.peft._base import AdapterBase, AdapterFactory
from bayescl.peft._clora import layer
from bayescl.peft._clora.config import CLoRAConfig
from bayescl.peft._create import iter_named_adapters


class CLoRA(AdapterFactory):
    """Continual LoRA (c-LoRA).

    .. [#f0] Smith, J. S., Hsu, Y.-C., Zhang, L., Hua, T., Kira, Z., Shen, Y., & Jin, H.
        (2024). Continual diffusion: Continual customization of text-to-image diffusion
        with c-LoRA. 2024. https://openreview.net/forum?id=TZdEgwZ6f3
    """

    @typechecked
    def __init__(self, config: CLoRAConfig = CLoRAConfig()):
        self.config = config

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(CLoRAConfig(**kwargs))

    def _get_replacement(self, module: nn.Module) -> AdapterBase | nn.Module:
        if isinstance(module, nn.Linear):
            return layer.CLoRALinear(
                module.in_features, module.out_features, config=self.config
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement

    @staticmethod
    def self_regularization_loss(module: nn.Module) -> Tensor:
        r"""Penalize changes to features learned by previous tasks.

        .. math::

            \mathcal{L}_{\text{reg}} =
            || \left| \sum_{t'=1}^{t-1} \mathbf{B}_{t'} \mathbf{A}_{t'} \right|
            \odot \mathbf{B}_t \mathbf{A}_t ||_F^2

        where :math:`\sum_{t'=1}^{t-1} \mathbf{B}_{t'} \mathbf{A}_{t'}` is pre-computed
        and stored in the adapter parameters as ``anchor_A`` and ``anchor_B``.

        :param module: The module containing the LoRA parameters.
        :return: The self-regularization loss.
        """

        loss = 0.0
        n = 0
        for name, sub_module in iter_named_adapters(module):
            if not isinstance(sub_module, layer.CLoRALayer):
                continue
            lhs = (sub_module.anchor_B @ sub_module.anchor_A).abs()
            rhs = sub_module.clora_B @ sub_module.clora_A
            loss += (lhs * rhs).norm(p="fro") ** 2
            n += 1
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("No loss accumulated, check if adapters were added.")
        return loss / n

    @staticmethod
    @torch.no_grad()
    def update_anchors(module: nn.Module) -> None:
        """Update the anchor parameters with the current LoRA parameters.

        This is used to update the anchors after training a new task.

        :param module: The module containing the LoRA parameters.
        """
        for name, sub_module in iter_named_adapters(module):
            if not isinstance(sub_module, layer.CLoRALayer):
                continue
            sub_module.anchor_A += sub_module.clora_A
            sub_module.anchor_B += sub_module.clora_B
            sub_module.reset_clora()
