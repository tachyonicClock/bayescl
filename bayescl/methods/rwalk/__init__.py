"""RWalk (Riemannian Walk / EWC++) over LoRA adapters.

Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian
Walk for Incremental Learning: Understanding Forgetting and Intransigence. ECCV.

This method is plain LoRA fine-tuning plus Avalanche's ``RWalkPlugin``.
"""

from ._arm import RWalk

__all__ = ["RWalk"]
