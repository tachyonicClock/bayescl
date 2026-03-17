from os import environ
from typing import Tuple

import jinja2
import optuna

STORAGE = optuna.storages.RDBStorage(environ.get("OPTUNA_STORAGE"))


MAX_TRIALS = 10

DATASETS = {
    "SplitCIFAR100": "cifar100",
    "SplitDomainNet": "domainnet",
    "SplitImageNetR": "imagenetr",
}

STUDIES = [
    "bayescl/hp/cifar100/sdlora",
    "bayescl/hp/core50/sdlora",
    "bayescl/hp/imagenetr/sdlora",
    "bayescl/hp/cifar100/clora",
    "bayescl/hp/core50/clora",
    "bayescl/hp/imagenetr/clora",
    "bayescl/hp/cifar100/inflora",
    "bayescl/hp/core50/inflora",
    "bayescl/hp/imagenetr/inflora",
    "bayescl/hp/cifar100/mas",
    "bayescl/hp/core50/mas",
    "bayescl/hp/imagenetr/mas",
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


def score(values: Tuple[float, float]) -> float:
    acc, ece = values
    return (acc + (1 - ece)) / 2


def main():
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
    jinja_template = jinja_env.get_template("script/template.jsonnet.jinja2")

    for study_name in STUDIES:
        try:
            study = optuna.load_study(study_name=study_name, storage=STORAGE)
        except KeyError:
            print(f"Study '{study_name}' not found, skipping...")
            continue

        _, _, dataset, method = study_name.split("/")

        trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ][:MAX_TRIALS]

        # Select the best trial based on the highest score (accuracy and ECE)
        best_trial = max(trials, key=lambda t: score(t.values))

        # score_ = score(best_trial.values)
        print("========================================")
        print(f"Dataset {dataset}, Method: {method}")
        print(f"Avg. Acc. {best_trial.values[0] * 100:.2f}")
        print(f"ECE       {best_trial.values[1] * 100:.2f}")
        print(f"Score:    {score(best_trial.values) * 100:.2f}")
        print(f"N Trials  {len(study.trials)}")

        git_hash = study.user_attrs.get("git_commit")

        config = {}

        for param in best_trial.params:
            keys = param.split(".")
            sub_config = config
            for key in keys[:-1]:
                sub_config = sub_config.setdefault(key, {})
            sub_config[keys[-1]] = best_trial.params[param]

        config_file = jinja_template.render(
            study_id=study._study_id,
            study_name=study_name,
            git_hash=git_hash,
            dataset=dataset,
            method=method,
            accuracy=best_trial.values[0],
            ece=best_trial.values[1],
            score=score(best_trial.values),
            n_trials=len(trials),
            hyperparameters=config,
        )
        print(config_file)

        filename = f"configs/{dataset}/{method}.jsonnet"
        with open(filename, "w") as f:
            f.write(config_file)


if __name__ == "__main__":
    main()
