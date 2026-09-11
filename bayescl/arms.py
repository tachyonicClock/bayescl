"""Import side-effect module: registers every arm.

Importing :mod:`bayescl.arms` populates :data:`bayescl.treatments._registry.ARMS`.
The CLI imports this before reading :func:`arm_names`.
"""

from bayescl.treatments._registry import ARMS, ArmBase, arm_names, get_arm, register
from bayescl.treatments.ball._arm import BALL
from bayescl.treatments.clora._arm import CLoRA
from bayescl.treatments.ewc._arm import EWC
from bayescl.treatments.inflora._arm import InfLoRA
from bayescl.treatments.lora._arm import LoRA
from bayescl.treatments.lora_ensemble._arm import LoRAEnsemble
from bayescl.treatments.lora_joint._arm import LoRAJoint
from bayescl.treatments.rwalk._arm import RWalk
from bayescl.treatments.sdlora._arm import SDLoRA
from bayescl.treatments.tball._arm import TBALL, TBALLMND

__all__ = [
    "ARMS",
    "ArmBase",
    "arm_names",
    "get_arm",
    "register",
    "BALL",
    "CLoRA",
    "EWC",
    "InfLoRA",
    "LoRA",
    "LoRAEnsemble",
    "LoRAJoint",
    "RWalk",
    "SDLoRA",
    "TBALL",
    "TBALLMND",
]
