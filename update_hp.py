import yaml
import glob
from pathlib import Path
from itertools import product
import optuna
from os import environ

STORAGE = environ.get("OPTUNA_STORAGE")
study_hash = "0ee8c6d"

DATASETS = {
    "cifar100": "SplitCIFAR100",
    "domainnet": "SplitDomainNet",
    "imagenetr": "SplitImageNetR",
}

METHODS = {
    "01_linear": "linear",
    "02_lora": "lora",
    "03_blob": "blob",
}




for (f_dataset, o_dataset), (f_method, o_method) in product(DATASETS.items(), METHODS.items()):
    filename = Path(f"configs/{f_dataset}/{f_method}.yaml")
    study_name = f"bayescl/{o_dataset}/{o_method}/{study_hash}"
    study = optuna.load_study(study_name=study_name, storage=STORAGE)

    best_trials = study.best_trials
    assert len(best_trials) == 1, "More than one best trial found!"
    best_trial = best_trials[0]
    print(filename)
    print(best_trial)

    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    for param in best_trial.params:
        keys = param.split(".")
        sub_config = config
        for key in keys[:-1]:
            sub_config = sub_config[key]
        sub_config[keys[-1]] = best_trial.params[param]

    with open(filename, "w") as f:
        yaml.dump(config, f)    
