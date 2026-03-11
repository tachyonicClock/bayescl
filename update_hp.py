from os import environ
from pathlib import Path
from typing import Tuple

import optuna
import yaml

STORAGE = environ.get("OPTUNA_STORAGE")

MAX_TRIALS = 10

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
    "l2p": "09_l2p",
    "tball": "11_tball",
}

STUDIES = [
    "bayescl/hp/cifar100/sdlora",
    # "bayescl/hp/cifar100/linear",
    # "bayescl/hp/cifar100/lora",
    # "bayescl/hp/cifar100/ball",
    # "bayescl/hp/cifar100/tball",
    # "bayescl/hp/cifar100/replay",
    # "bayescl/hp/cifar100/gdumb",
    # "bayescl/hp/cifar100/der",
    # "bayescl/hp/cifar100/joint",
    # "bayescl/hp/cifar100/rwalk",
    # "bayescl/hp/cifar100/l2p",
    # "bayescl/hp/cifar100/tball",
    # "bayescl/hp/imagenetr/linear",
    # "bayescl/hp/imagenetr/lora",
    # "bayescl/hp/imagenetr/ball",
    # "bayescl/hp/imagenetr/replay",
    # "bayescl/hp/imagenetr/gdumb",
    # "bayescl/hp/imagenetr/der",
    # "bayescl/hp/imagenetr/joint",
    # "bayescl/hp/imagenetr/rwalk",
    # "bayescl/hp/imagenetr/l2p",
    # "bayescl/hp/imagenetr/tball",
    # "bayescl/hp/core50/linear",
    # "bayescl/hp/core50/lora",
    # "bayescl/hp/core50/ball",
    # "bayescl/hp/core50/replay",
    # "bayescl/hp/core50/gdumb",
    # "bayescl/hp/core50/der",
    # "bayescl/hp/core50/joint",
    # "bayescl/hp/core50/rwalk",
    # "bayescl/hp/core50/l2p",
    # "bayescl/hp/core50/tball",
]


# Set the precision of the float numbers in the YAML file to two significant figures
def float_representer(dumper, value):
    text = f"{value:.2g}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


def score(values: Tuple[float, float]) -> float:
    acc, ece = values
    return (acc + (1 - ece)) / 2


yaml.add_representer(float, float_representer)


for study_name in STUDIES:
    _, _, f_dataset, o_method = study_name.split("/")
    # f_dataset = DATASETS[o_dataset]
    filename = Path(f"configs/{f_dataset}/{o_method}.yaml")

    try:
        study = optuna.load_study(study_name=study_name, storage=STORAGE)
    except KeyError:
        print(f"Study '{study_name}' not found, skipping...")
        continue

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE][
        :MAX_TRIALS
    ]

    # select the trial with the best ECE (second value in the values tuple)
    best_trial = max(trials, key=lambda t: score(t.values))

    score_ = score(best_trial.values)
    print("========================================")
    print(f"Dataset {f_dataset}, Method: {o_method}")
    print(f"Avg. Acc. {best_trial.values[0] * 100:.2f}")
    print(f"ECE       {best_trial.values[1] * 100:.2f}")
    print(f"Score:    {score_}")
    print(f"N Trials  {len(study.trials)}")

    config = {
        "include": [
            "../base.yaml",
            f"../base/dataset/{f_dataset}.yaml",
            f"../base/method/{o_method}.yaml",
        ]
    }

    for param in best_trial.params:
        keys = param.split(".")
        sub_config = config
        for key in keys[:-1]:
            sub_config = sub_config.setdefault(key, {})
        sub_config[keys[-1]] = best_trial.params[param]

    git_hash = study.user_attrs.get("git_commit")

    with open(filename, "w") as f:
        f.writelines(
            [
                f"# {study_name} {git_hash}\n",
                f"# {best_trial.values[0] * 100:.2f}% Acc. {best_trial.values[1] * 100:.2f}% ECE\n",
                f"# Score {score_ * 100:.2f}% (ACC+(1-ECE))/2\n",
                f"# Selected best run based on highest score {len(trials)} trials\n",
            ]
        )
        yaml.dump(config, f)
