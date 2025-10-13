import os

import click
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns


@click.argument("study_name")
@click.option("--hue", default="lr", help="Parameter to use for hue in pairplot")
@click.command()
def main(study_name: str, hue: str):
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
    

    # sns.set_theme(style="ticks", context="paper")
    hue_key = f"params_{hue}"
    grid = sns.pairplot(
        df,
        x_vars=[col for col in df.columns if col.startswith("params_")],
        y_vars=["Acc", "ECE"],
        hue=hue_key if hue_key in df.columns else None,
        height=2.0
    )

    # Set labels in pairplot
    for ax in grid.axes.flatten():
        if ax is not None:
            if ax.get_xlabel().startswith("params_"):
                label = ax.get_xlabel()[len("params_") :]
                ax.set_xlabel(label.split(".")[-1])


    for x_var in grid.x_vars:
        # get search space
        if x_var.startswith("params_"):
            key = x_var[len("params_") :]
            distributions = study.trials[-1].distributions
            if key not in distributions:
                continue
            dist = distributions[key]
            if isinstance(dist, optuna.distributions.FloatDistribution):
                if dist.log:
                    grid.axes[0, grid.x_vars.index(x_var)].set_xscale("log")

    grid.figure.suptitle(study_name + f" ('{attr_git_message}' {attr_git_commit})", y=1.02, fontsize=10)
    grid.figure.savefig(f"figs/{study_name.replace('/', '-')}.png", dpi=300, bbox_inches="tight")

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
