from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Sequence, Union

import optuna
import pydantic


class _BaseConfig(pydantic.BaseModel):
    model_config: pydantic.ConfigDict = {"extra": "forbid"}  # type: ignore


class Categorical(_BaseConfig):
    #: The type controls how the hyperparameter is sampled.
    type: Literal["categorical"] = "categorical"
    #: Parameter value candidates.
    choices: Sequence[None | bool | int | float | str]


class Float(_BaseConfig):
    #: The type controls how the hyperparameter is sampled.
    type: Literal["float"] = "float"
    #: Lower endpoint of the range of suggested values.
    low: float
    #: Upper endpoint of the range of suggested values.
    high: float
    #: A step of discretization.
    step: Optional[float] = None
    #: A flag to sample the value from the log domain or not.
    log: bool = False


class Int(_BaseConfig):
    #: The type controls how the hyperparameter is sampled.
    type: Literal["int"] = "int"
    #: Lower endpoint of the range of suggested values
    low: int
    #: Upper endpoint of the range of suggested values.
    high: int
    #: A step of discretization.
    step: int = 1
    #: A flag to sample the value from the log domain or not.
    log: bool = False


class Constant(_BaseConfig):
    #: The type controls how the hyperparameter is sampled.
    type: Literal["constant"] = "constant"
    #: The constant value to be used.
    value: None | bool | int | float | str


SearchSpace = Annotated[
    Union[Categorical, Float, Int, Constant], pydantic.Field(discriminator="type")
]


class PrunerType(str, Enum):
    NopPruner = "NopPruner"
    MedianPruner = "MedianPruner"
    HyperbandPruner = "HyperbandPruner"
    SuccessiveHalvingPruner = "SuccessiveHalvingPruner"


class SamplerType(str, Enum):
    RandomSampler = "RandomSampler"
    TPESampler = "TPESampler"
    QMCSampler = "QMCSampler"
    BruteForceSampler = "BruteForceSampler"


class HyperparameterSearch(_BaseConfig):
    #: hyperparameters to search for. Keys should use dot notation "key.subkey.subsubkey"
    params: Dict[str, SearchSpace]
    #: Direction of optimization.
    direction: Sequence[Literal["maximize", "minimize"]] = ["minimize"]
    #: The number of trials
    n_trials: int
    #: The hyperparameter optimization algorithm to use.
    sampler: SamplerType = SamplerType.QMCSampler
    #: Pruner to use. NopPruner disables pruning.
    pruner: PrunerType = PrunerType.NopPruner


def get_pruner(pruner: PrunerType) -> "optuna.pruners.BasePruner":
    if pruner == PrunerType.NopPruner:
        return optuna.pruners.NopPruner()
    elif pruner == PrunerType.MedianPruner:
        return optuna.pruners.MedianPruner()
    elif pruner == PrunerType.HyperbandPruner:
        return optuna.pruners.HyperbandPruner()
    elif pruner == PrunerType.SuccessiveHalvingPruner:
        return optuna.pruners.SuccessiveHalvingPruner()
    raise ValueError(f"Unknown pruner: {pruner}")


def get_sampler(sampler: SamplerType) -> "optuna.samplers.BaseSampler":
    if sampler == SamplerType.TPESampler:
        return optuna.samplers.TPESampler()
    elif sampler == SamplerType.RandomSampler:
        return optuna.samplers.RandomSampler()
    elif sampler == SamplerType.BruteForceSampler:
        return optuna.samplers.BruteForceSampler()
    elif sampler == SamplerType.QMCSampler:
        return optuna.samplers.QMCSampler(scramble=True)
    raise ValueError(f"Unknown sampler: {sampler}")


def obj_dot_notation_set(key: str, obj: object, value: Any) -> object:
    root = obj
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return root


def optuna_suggest(trial: Any, config: Any, params: Dict[str, SearchSpace]) -> Any:
    import optuna

    assert isinstance(trial, optuna.Trial)

    for name, hp in params.items():
        if isinstance(hp, Categorical):
            value = trial.suggest_categorical(name, hp.choices)
        elif isinstance(hp, Float):
            value = trial.suggest_float(name, hp.low, hp.high, step=hp.step, log=hp.log)
        elif isinstance(hp, Int):
            value = trial.suggest_int(name, hp.low, hp.high, step=hp.step, log=hp.log)
        elif isinstance(hp, Constant):
            value = hp.value
        else:
            raise RuntimeError(f"Unknown hp suggestion {hp}")
        obj_dot_notation_set(name, config, value)
    return config
