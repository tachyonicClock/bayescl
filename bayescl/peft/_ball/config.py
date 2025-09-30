from dataclasses import dataclass, field

from bayescl.vbnn import VBNNConfig


@dataclass
class BALLConfig:
    r: int = 4
    lora_alpha: int = 1
    vbnn: VBNNConfig = field(default_factory=VBNNConfig)
    # prior_mean: float = 0.0
    # prior_weight_sd: float = 1.0
    # prior_bias_sd: float = 1.0
    # init_sd: float = 1e-4
    # max_sd: Optional[float] = None
    # local_reparameterization: bool = True
    # nonlinearity_scale: float = 1.0
    # sqrt_width_scaling: bool = False
