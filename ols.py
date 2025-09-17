from os import environ

import numpy as np
import optuna
import tabulate

n = 100

target_trial_count = 15


def sec_to_hh_mm_ss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


OPTUNA_STORAGE = environ.get("OPTUNA_STORAGE")

study_names = optuna.get_all_study_names(storage=OPTUNA_STORAGE)[-100:]
study_names = reversed(study_names)  # most recent first


rows = []

for study_name in study_names:
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
    trials_complete = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    trials_running = [
        t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
    ]

    durations = [t.duration.total_seconds() for t in trials_complete]
    avg_duration_s = np.mean(durations) if len(durations) > 0 else 0.0
    std_duration_s = np.std(durations) if len(durations) > 0 else 0.0

    n_complete = len(trials_complete)
    n_running = len(trials_running)
    n_total = len(study.trials)

    complete_in = 0
    if n_complete < target_trial_count and avg_duration_s > 0:
        complete_in = (
            (target_trial_count - n_complete) * avg_duration_s / max(1, n_running)
        )

    rows.append(
        {
            "Study ID": study._study_id,
            "Study Name": study.study_name,
            "N Complete": len(trials_complete),
            "N Running": len(trials_running),
            "N Total": len(study.trials),
            "Mean Duration": sec_to_hh_mm_ss(avg_duration_s),
            "Remaining": sec_to_hh_mm_ss(complete_in),
        }
    )

print(tabulate.tabulate(rows, headers="keys"))
