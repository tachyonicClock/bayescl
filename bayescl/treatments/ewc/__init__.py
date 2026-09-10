"""EWC (Elastic Weight Consolidation) over LoRA adapters.

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural
networks. PNAS. https://www.pnas.org/doi/10.1073/pnas.1611835114

This method is plain LoRA fine-tuning plus Avalanche's online ``EWCPlugin``.
"""

from ._arm import EWC

__all__ = ["EWC"]
