import os
import sys
from os import environ

sys.path.append(os.getcwd())

import optuna

from bayescl.config import from_configs
from bayescl.experiment import Experiment


def objective(trial: optuna.Trial) -> float:
    # Constant parameters
    config = from_configs(["configs/SplitCIFAR100/blob.yaml"])
    assert config.peft and config.peft.type == "BLoB"

    config.study_name = trial.study.study_name

    # Nuisance parameters
    config.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    # Scientific parameters
    config.peft.beta = trial.suggest_float("config.peft.beta", 1, 5_000)
    config.peft.config.blob_A.prior_sigma = trial.suggest_float(
        "A_prior_sigma", 0.01, 5.0
    )
    config.peft.config.blob_B.prior_sigma = trial.suggest_float(
        "B_prior_sigma", 0.01, 5.0
    )
    return Experiment(config).run()


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="BLoB_02",
        storage=environ.get("OPTUNA_STORAGE"),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
