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


@dataclass(kw_only=True)
class BALLConfig:
    class PerturbationType(Enum):
        ADDITIVE = "additive"
        """LoRA-style additive perturbation."""
        MULTIPLICATIVE = "multiplicative"
        """LoRMA/Rank-1 BNN-style multiplicative perturbation.
        
        Bihany, Harsh, Shubham Patel, and Ashutosh Modi. “LoRMA: Low-Rank Multiplicative
        Adaptation for LLMs.” In Findings of the Association for Computational
        Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, edited by
        Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar.
        Association for Computational Linguistics, 2025.
        https://aclanthology.org/2025.findings-acl.527/.
        """

    r: int = 4
    lora_alpha: int = 1
    ensemble_size: int = 1
    """Number of ensemble members in the Batch-Ensemble."""
    vbnn: VBNNConfig = field(default_factory=VBNNConfig)
    """Configuration for the underlying Bayesian Neural Network."""
    dropout: float = 0.0
    """Dropout rate to use on the adapter inputs."""
    perturbation_type: PerturbationType = PerturbationType.ADDITIVE
    """Type of perturbation to use in the Batch-Ensemble adapters."""
