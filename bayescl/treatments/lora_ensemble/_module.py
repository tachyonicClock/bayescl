import torch
import torch.nn as nn


class EnsembleModule(nn.Module):
    """A collection of independently parameterised member models.

    Training and evaluation are driven by ``EnsembleStrategy``, which reads
    ``members`` directly to get per-member gradients/optimizers. ``forward``
    is only used for anything that calls the module directly (e.g. parameter
    summaries), and mirrors the eval-time combination rule: average
    per-member softmax probabilities, then return log-probabilities so
    downstream code that expects logits (softmax + argmax) still works.
    """

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([member(x) for member in self.members])
        mean_probs = outputs.softmax(dim=-1).mean(dim=0)
        return mean_probs.clamp_min(1e-12).log()
