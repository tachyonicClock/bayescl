import argparse
from itertools import product
from os import environ

import numpy as np
import optuna
import tabulate

n = 100

target_trial_count = 15


SCENARIOS = [
    "cifar100",
    "imagenetr",
    "domainnet",
]

METHODS = [
    "linear",
    "lora",
    "ball",
    "replay",
    "gdumb",
    "der",
    "joint",
    "rwalk",
]


def sec_to_hh_mm_ss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def main(prefix: str | None = None, sort_on_ece: bool = False):
    OPTUNA_STORAGE = environ.get("OPTUNA_STORAGE")

    study_names = optuna.get_all_study_names(storage=OPTUNA_STORAGE)
    # study_names = reversed(study_names)  # most recent first

    rows = []

    for scenario, method in product(SCENARIOS, METHODS):
        study_name = f"{prefix}/{scenario}/{method}"

        if study_name not in study_names:
            continue

        study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
        trials_complete = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trials_running = [
            t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
        ]

        if len(trials_complete) == 0:
            continue

        try:
            if sort_on_ece:
                best_trial = min(study.best_trials, key=lambda t: t.values[1])
            else:
                best_trial = max(study.best_trials, key=lambda t: t.values[0])
        except IndexError:
            continue

        if len(best_trial.values) != 2:
            continue

        durations = [t.duration.total_seconds() for t in trials_complete]
        avg_duration_s = np.mean(durations) if len(durations) > 0 else 0.0
        std_duration_s = np.std(durations) if len(durations) > 0 else 0.0

        n_complete = len(trials_complete)
        n_running = len(trials_running)
        # n_total = len(study.trials)

        # complete_in = 0
        # if n_complete < target_trial_count and avg_duration_s > 0:
        #     complete_in = (
        #         (target_trial_count - n_complete) * avg_duration_s / max(1, n_running)
        #     )

        rows.append(
            {
                # "Study ID": study._study_id,
                "Study Name": study.study_name[len(prefix) + 1 :],
                "done/total (running)": f"{n_complete}/{target_trial_count} ({n_running})",
                # "N best": len(study.best_trials),
                # "N Running": ,
                # "N Total": len(study.trials),
                "Mean Time": sec_to_hh_mm_ss(avg_duration_s),
                # "Std Time": sec_to_hh_mm_ss(std_duration_s),
                # "Remaining": sec_to_hh_mm_ss(complete_in),
                "Acc.": f"{best_trial.values[0] * 100:.2f}",
                "ECE": f"{best_trial.values[1] * 100:.2f}",
            }
        )

    print(tabulate.tabulate(rows, headers="keys"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="bayescl",
        help="Prefix of the study names to filter",
    )
    # sort on ece instead of accuracy
    parser.add_argument(
        "--ece",
        action="store_true",
        help="Sort the output based on ECE instead of accuracy",
    )

    args = parser.parse_args()
    main(args.prefix, args.ece)
