from os import environ

import optuna

from bayescl.config import from_configs
from bayescl.experiment import Experiment


def objective(trial: optuna.Trial) -> float:
    config = from_configs(["configs/SplitCIFAR100/blob.yaml"])
    config.train_epochs = 1
    config.max_tasks = 1
    assert config.peft and config.peft.type == "BLoB"
    config.peft.beta = 0.0
    config.peft.config.blob_A.init_sigma_mean = trial.suggest_float(
        "init_sigma_mean", 1e-5, 0.1
    )
    config.peft.config.blob_A.init_rho_std = trial.suggest_float(
        "init_rho_std", 1e-5, 0.1
    )
    return Experiment(config).run()


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="bayescl/BLoB_A_initialization",
        storage=environ.get("OPTUNA_STORAGE"),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
