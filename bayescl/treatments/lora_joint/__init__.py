"""LoRA-Joint: a deterministic, non-continual upper-bound baseline.

At every task boundary the model is retrained on all classes seen so far
(Avalanche's ``Cumulative`` strategy), rather than only the current task's
data. This stays causally valid across evaluation points -- unlike training
once on the full stream upfront -- since it never sees future-task data.
"""

from ._arm import LoRAJoint

__all__ = ["LoRAJoint"]
