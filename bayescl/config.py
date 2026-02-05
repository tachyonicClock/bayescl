from os import environ
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from bayescl.util.optuna import HyperparameterSearch
from bayescl.vbnn import VBNNConfig


class BaseConfig(BaseModel):
    model_config: ConfigDict = {"extra": "forbid"}  # type: ignore

    def kwargs(self) -> dict:
        return self.model_dump(exclude={"type"})  # type: ignore


# --- Scenario Configurations ---


class Scenario(BaseConfig):
    dataset: Literal[
        "MNIST",
        "CIFAR10",
        "CIFAR100",
        "ImageNetR",
        "DomainNet",
    ] = "MNIST"
    n_tasks: int = 5
    shuffle: bool = True
    #: Replace the test set with a validation set
    validation: bool = False


# --- Model Configurations ---


class BasicModelConfig(BaseConfig):
    type: Literal["basic"] = "basic"
    name: Literal["SimpleMLP", "SimpleFCGMLP"] = "SimpleMLP"


class HuggingFaceModelConfig(BaseConfig):
    type: Literal["huggingface"] = "huggingface"
    name: str = "facebook/dinov2-small"
    freeze_backbone: bool = True
    #: If adapters are used, regex to filter which modules to add adapters to.
    adapter_filter: str
    head_module: str


# --- Plugin Configurations ---


class OutlierExposureConfig(BaseModel):
    strength: float = 1.0
    batch_size: Optional[int] = None


class EWCConfig(BaseConfig):
    ewc_lambda: float
    decay_factor: float


class RWalkConfig(BaseModel):
    ewc_lambda: float
    """hyperparameter to weigh the penalty inside the total loss. The larger the lambda,
    the larger the regularization."""
    ewc_alpha: float
    """Specify the moving average factor for the importance matrix, as defined RWalk
    paper (a.k.a. EWC++). Higher values lead to higher weight to newly computed
    importances. Must be in [0, 1]. Defaults to 0.9.
    """
    delta_t: int
    """Specify the iterations interval in which the parameter scores are updated."""


# --- PEFT Configurations ---


class LoRAConfig(BaseConfig):
    type: Literal["LoRA"] = "LoRA"
    r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.0


class L2PConfig(BaseConfig):
    type: Literal["L2P"] = "L2P"
    pull_constraint_coeff: float
    prompts_per_task: int
    prompt_length: int
    top_k: int


class BALLConfig(BaseConfig):
    type: Literal["BALL"] = "BALL"
    r: int = 4
    """Rank of the LoRA adapters."""
    lora_alpha: int = 1
    """Scaling factor for the LoRA adapters."""
    dropout: float = 0.0
    """Dropout rate to use on the adapter inputs."""
    vbnn: VBNNConfig
    """Configuration for the underlying Bayesian Neural Network."""


class TBALLConfig(BaseConfig):
    type: Literal["TBALL"] = "TBALL"
    rank: int
    """Rank of the TBALL adapters."""
    alpha: float = 1.0
    """Scaling factor for the TBALL adapters."""
    prior_mean: float = 0.0
    """Prior mean for the Bayesian layers."""
    prior_weight_sd: float = 1.0
    """Prior standard deviation for the weights."""
    init_sd: float = 1e-4
    """Standard deviation for initializing the variational parameters."""
    nonlinearity_scale: float = 1.0
    """Scale for the nonlinearity in the Bayesian layers."""
    bnn: Literal["FCG", "FFG"] = "FCG"
    """Full covariance (FCG) or fully factorized Gaussian (FFG) Bayesian layers."""
    bias: bool = False
    """Whether to include bias in the Bayesian layers."""


PEFTConfig = LoRAConfig | L2PConfig | BALLConfig | TBALLConfig

# --- Strategy Configurations ---


class NaiveConfig(BaseConfig):
    type: Literal["Naive"] = "Naive"


class DERConfig(BaseConfig):
    type: Literal["DER"] = "DER"
    mem_size: int
    """Fixed memory size"""
    alpha: float
    """Hyperparameter weighting the MSE loss"""
    beta: float
    """Hyperparameter weighting the CE loss, when more than 0, DER++ is used instead of 
    DER"""


class GDumbConfig(BaseConfig):
    type: Literal["GDumb"] = "GDumb"
    mem_size: int
    """Fixed memory size"""


class VCLConfig(BaseConfig):
    """Variational Continual Learning Training Strategy.

    Requires a model or a PEFT module that implements Bayesian layers.
    """

    type: Literal["VCL"] = "VCL"
    beta: float = 1.0
    """Hyperparameter weighting the KL divergence loss."""
    train_samples: int
    """Number of samples for each step of training."""
    test_samples: int
    """Number of samples for each step of testing."""
    softmax_avg: bool = False
    """If true, softmax then average, otherwise average then softmax."""


StrategyConfig = NaiveConfig | DERConfig | GDumbConfig | VCLConfig


class Label(BaseConfig):
    """Label given to the experiment: ``{study}/{scenario}/{method}/{run}``"""

    study: str = "manual"
    """Top level label for a group of related experiments."""
    scenario: str = "unlabeled_scenario"
    """Label for the scenario or dataset used."""
    method: str = "unlabeled_method"
    """Label for the method or strategy used."""
    run: Optional[str] = None
    """Label for the individual run, e.g. 0001 for hyperparameter search trials."""


class Config(BaseConfig):
    include: list[Path] = []

    label: Label = Field(Label())

    #: Scenario configuration.
    scenario: Scenario = Field(Scenario())

    #: Model configuration.
    model: BasicModelConfig | HuggingFaceModelConfig | L2PConfig = Field(
        BasicModelConfig(),
        discriminator="type",
    )

    peft: Optional[PEFTConfig] = Field(None, discriminator="type")
    """Optional parameter-efficient fine-tuning configuration."""

    #: Parent directory containing datasets.
    dataset_root: str = environ.get("DATASETS", "./datasets")
    #: Parent directory containing run logs.
    log_root: str = "./log"

    # Strategy
    #: Device to use for training (cuda or cpu)
    device: str = "cuda"
    #: Learning rate for the optimizer
    lr: float = 0.001
    #: Mini-batch size for training
    train_mb_size: int = 500
    #: Mini-batch size for evaluation. If None, defaults to train_mb_size
    eval_mb_size: Optional[int] = None
    #: Number of epochs for training each experience
    epochs: int = 1
    #: Evaluate every N epochs. If -1, only evaluate at the end of each experience
    eval_every: int = -1

    #: Number of epochs in the first experience
    first_exp_epochs: Optional[int] = None

    #: Add the outlier exposure plugin
    outlier_exposure: Optional[OutlierExposureConfig] = None

    #: Add the RWalk plugin
    rwalk: Optional[RWalkConfig] = None

    #: Number of workers for data loading
    num_workers: int = 0

    strategy: StrategyConfig = Field(NaiveConfig(), discriminator="type")
    """If None use naive strategy with configured plugins. If specified, use the given
    strategy with the configured plugins.
    """

    #: Number of samples in the replay memory
    replay: int = 0

    #: Stop after training on this many tasks
    max_tasks: Optional[int] = None

    #: Use local cross entropy by masking the output layer during training
    use_local_ce: bool = True

    ewc: Optional[EWCConfig] = None
    """If not None, use EWC with the given configuration."""

    hpsearch: Optional[HyperparameterSearch] = None

    #: Random seed for reproducibility
    seed: int = 0

    checkpoint: bool = False
    """Whether to save a checkpoint after each experience/task."""


def _resolve_includes(base: Path, filenames: list[str]) -> DictConfig:
    if filenames:
        logger.info(f"Including configs {filenames}")
        config = OmegaConf.merge(*(OmegaConf.load(base / f) for f in filenames))
        assert config.get("include") is None, "Nested includes are not supported"
        return config  # type: ignore
    else:
        return DictConfig({})


def from_configs(
    config_filenames: list[str], dotlist: list[str] | None = None
) -> Config:
    config = DictConfig({})  # type: ignore
    for file in config_filenames:
        file = Path(file)
        logger.info(f"Updating config from {file}")
        config = OmegaConf.merge(config, OmegaConf.load(file))  # type: ignore
        included = _resolve_includes(file.parent, config.get("include", []))  # type: ignore
        config = OmegaConf.merge(included, config)
    if dotlist is not None:
        logger.info(f"Updating config from command line args: {dotlist}")
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))

    return Config.model_validate(OmegaConf.to_object(config))
