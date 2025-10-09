from os import environ
from pathlib import Path

import optuna
import yaml

STORAGE = environ.get("OPTUNA_STORAGE")

DATASETS = {
    "SplitCIFAR100": "cifar100",
    "SplitDomainNet": "domainnet",
    "SplitImageNetR": "imagenetr",
}

METHODS = {
    "linear": "01_linear",
    "lora": "02_lora",
    "ball": "03_ball",
    "replay": "04_replay",
    "gdumb": "05_gdumb",
    "der": "06_der",
    "joint": "07_joint",
    "rwalk": "08_rwalk",
}

STUDIES = [
    "bayescl/hp_0100/cifar100/ball",
    "bayescl/hp_0100/domainnet/linear",
    "bayescl/hp_0100/domainnet/lora",
    # "bayescl/hp_0100/domainnet/ball",
    "bayescl/hp_0100/domainnet/replay",
    # "bayescl/hp_0100/domainnet/gdumb",
    "bayescl/hp_0100/domainnet/der",
    # "bayescl/hp_0100/domainnet/joint",
    "bayescl/hp_0100/domainnet/rwalk",
]


# Set the precision of the float numbers in the YAML file to two significant figures
def float_representer(dumper, value):
    text = f"{value:.2g}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


yaml.add_representer(float, float_representer)


for study_name in STUDIES:
    _, _, f_dataset, o_method = study_name.split("/")
    # f_dataset = DATASETS[o_dataset]
    f_method = METHODS[o_method]
    filename = Path(f"configs/{f_dataset}/{f_method}.yaml")

    try:
        study = optuna.load_study(study_name=study_name, storage=STORAGE)
    except KeyError:
        print(f"Study '{study_name}' not found, skipping...")
        continue

    best_trials = study.best_trials
    # select the trial with the best ECE (second value in the values tuple)
    best_trial = min(best_trials, key=lambda t: t.values[1])

    print("========================================")
    print(f"Dataset {f_dataset}, Method: {o_method}")
    print(f"Avg. Acc. {best_trial.values[0] * 100:.2f}")
    print(f"ECE       {best_trial.values[1] * 100:.2f}")
    print(f"N Trials  {len(study.trials)}")

    config = {
        "include": [
            "../base.yaml",
            f"../base/dataset/{f_dataset}.yaml",
            f"../base/method/{f_method}.yaml",
        ]
    }

    for param in best_trial.params:
        keys = param.split(".")
        sub_config = config
        for key in keys[:-1]:
            sub_config = sub_config.setdefault(key, {})
        sub_config[keys[-1]] = best_trial.params[param]

    with open(filename, "w") as f:
        yaml.dump(config, f)
