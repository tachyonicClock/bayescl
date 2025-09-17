from itertools import product
from os import environ
from pathlib import Path

import optuna
import yaml

STORAGE = environ.get("OPTUNA_STORAGE")
study_hash = "870a71c"

DATASETS = {
    "cifar100": "SplitCIFAR100",
    "domainnet": "SplitDomainNet",
    "imagenetr": "SplitImageNetR",
}

METHODS = {
    "01_linear": "linear",
    # "02_lora": "lora",
    "03_ball": "ball",
}


# Set the precision of the float numbers in the YAML file to two significant figures
def float_representer(dumper, value):
    text = f"{value:.2g}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


yaml.add_representer(float, float_representer)


for (f_dataset, o_dataset), (f_method, o_method) in product(
    DATASETS.items(), METHODS.items()
):
    filename = Path(f"configs/{f_dataset}/{f_method}.yaml")
    study_name = f"bayescl/{o_dataset}/{o_method}/{study_hash}"
    try:
        study = optuna.load_study(study_name=study_name, storage=STORAGE)
    except KeyError:
        print(f"Study '{study_name}' not found, skipping...")
        continue

    best_trials = study.best_trials
    # select the trial with the best ECE (second value in the values tuple)
    best_trial = min(best_trials, key=lambda t: t.values[1])

    print("========================================")
    print(f"Dataset {o_dataset}, Method: {o_method}")
    print(f"Avg. Acc. {best_trial.values[0] * 100:.2f}")
    print(f"ECE       {best_trial.values[1] * 100:.2f}")
    print(f"N Trials  {len(study.trials)}")

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
