from dataclasses import dataclass, field
from enum import Enum

from bayescl.vbnn import VBNNConfig


class VarianceReduction(Enum):
    """Bayesian Neural Networks are probabilistic models, and thus have inherent
    variance in their predictions. This can make training with gradient descent
    difficult. Several variance reduction techniques have been proposed to mitigate this
    issue.
    """

    LRT = "lrt"
    """Use the Local Reparameterization Trick estimator.

    Sample different weights for each input in a mini-batch.
    
    Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational dropout and the local
    reparameterization trick. https://arxiv.org/abs/1506.02557
    """
    FLIPOUT = "flipout"
    """Enable the Flipout estimator.
    
    Wen, Y., Vicol, P., Ba, J., Tran, D., & Grosse, R. B. (2018). Flipout: Efficient
    pseudo-independent weight perturbations on mini-batches. 6th International
    Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 -
    May 3, 2018, Conference Track Proceedings. https://openreview.net/forum?id=rJNpifWAb
    """
    NONE = "none"
    """No variance reduction"""


@dataclass
class BALLConfig:
    r: int = 4
    lora_alpha: int = 1
    mode: VarianceReduction = VarianceReduction.NONE
    """Variance reduction method to use. Changes how the forward pass is computed in training mode."""
    vbnn: VBNNConfig = field(default_factory=VBNNConfig)
    # prior_mean: float = 0.0
    # prior_weight_sd: float = 1.0
    # prior_bias_sd: float = 1.0
    # init_sd: float = 1e-4
    # max_sd: Optional[float] = None
    # local_reparameterization: bool = True
    # nonlinearity_scale: float = 1.0
    # sqrt_width_scaling: bool = False
