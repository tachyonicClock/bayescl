from dataclasses import dataclass


@dataclass
class CLoRAConfig:
    r: int = 4
    lora_alpha: int = 1
