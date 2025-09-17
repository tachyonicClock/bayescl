from dataclasses import dataclass, field

from bayescl.vbnn import VBNNConfig


@dataclass
class BALLConfig:
    r: int = 4
    lora_alpha: int = 1
    vbnn: VBNNConfig = field(default_factory=VBNNConfig)
