"""Import side-effect module: registers every arm.

Importing :mod:`bayescl.arms` populates :data:`bayescl.methods._registry.ARMS`.
The CLI imports this before reading :func:`arm_names`.
"""

from bayescl.methods._registry import ARMS, ArmBase, arm_names, get_arm, register
from bayescl.methods.ball._arm import BALL
from bayescl.methods.clora._arm import CLoRA
from bayescl.methods.ewc._arm import EWC
from bayescl.methods.inflora._arm import InfLoRA
from bayescl.methods.lora._arm import LoRA
from bayescl.methods.rwalk._arm import RWalk
from bayescl.methods.sdlora._arm import SDLoRA
from bayescl.methods.tball._arm import TBALL, TBALLMND

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
    "RWalk",
    "SDLoRA",
    "TBALL",
    "TBALLMND",
]
