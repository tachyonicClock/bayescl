from os import environ

import optuna

from bayescl.config import from_configs
from bayescl.experiment import Experiment


def objective(trial: optuna.Trial) -> float:
    config = from_configs(["configs/SplitCIFAR100/blob.yaml"])
    assert config.peft and config.peft.type == "BLoB"
    config.peft.beta = trial.suggest_float("beta", 1e-4, 0.1, log=True)
    return Experiment(config).run()


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="bayescl/BLoB_beta_02",
        storage=environ.get("OPTUNA_STORAGE"),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
