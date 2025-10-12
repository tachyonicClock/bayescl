import os

import click
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns


@click.argument("study_name")
@click.command()
def main(study_name: str):
    study = optuna.load_study(
        study_name=study_name, storage=os.environ["OPTUNA_STORAGE"]
    )
    attr_git_commit = study.user_attrs.get("git_commit", "unknown")
    attr_git_message = study.user_attrs.get("git_message", "unknown")

    df = study.trials_dataframe()
    # rename values_0 to train and values_1 to val
    df = df.rename(columns={"values_0": "Acc", "values_1": "ECE"})
    # save to csv
    df.to_csv(f"figs/{study_name.replace('/', '-')}.csv", index=False)

    sns.set_theme(style="ticks", context="paper")
    grid = sns.pairplot(
        df,
        y_vars=["Acc", "ECE"],
        # hue="params_peft.config.vbnn.sd_mode",
    )

    for x_var in grid.x_vars:
        # get search space
        if x_var.startswith("params_"):
            distributions = study.trials[0].distributions
            dist = distributions[x_var[len("params_") :]]
            if isinstance(dist, optuna.distributions.FloatDistribution):
                if dist.log:
                    grid.axes[0, grid.x_vars.index(x_var)].set_xscale("log")

    grid.figure.suptitle(study_name + f" ('{attr_git_message}' {attr_git_commit})")
    grid.figure.savefig(f"figs/{study_name.replace('/', '-')}.png")

    # plot hyperparameter importance
    importance_eval = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
    acc_importance = optuna.importance.get_param_importances(
        study, evaluator=importance_eval, target=lambda t: t.values[0]
    )
    ece_importance = optuna.importance.get_param_importances(
        study, evaluator=importance_eval, target=lambda t: t.values[1]
    )

    ratio = 2 / 1
    width = 4
    fig, ax = plt.subplots(figsize=(width, width / ratio))
    df_acc = pd.DataFrame.from_dict(
        acc_importance, orient="index", columns=["Importance"]
    ).reset_index()
    df_acc["Metric"] = "Acc"
    df_ece = pd.DataFrame.from_dict(
        ece_importance, orient="index", columns=["Importance"]
    ).reset_index()
    df_ece["Metric"] = "ECE"
    df_importance = pd.concat([df_acc, df_ece])
    sns.barplot(data=df_importance, y="index", x="Importance", hue="Metric", ax=ax)
    ax.set_title(
        study_name + f"\n('{attr_git_message}' {attr_git_commit})", fontsize=10
    )
    ax.set_xlabel("Importance")
    # remove y label
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"figs/{study_name.replace('/', '-')}_importance.png", dpi=300)


if __name__ == "__main__":
    main()
