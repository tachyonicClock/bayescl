DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]
METHODS = [
    # "ball",
    # "ewc",
    # "joint",
    # "linear",
    # "lora",
    # "mas",
    # "rwalk",
    # "tball",
    "clora",
    "sdlora",
]
N_RUNS = 5


def run_string(trial: int, dataset: str, method: str):
    label = f"{dataset[:5]}_{method[:5]}"
    cli = [
        f"ts -G 1 -L {label:<11}",
        "notirun.sh",
        "python main.py",
        f"-c configs/{dataset}/{method}.jsonnet".rjust(35),
        "-a checkpoint=True" if method.endswith("ball") else "",
        f"run test {trial}",
    ]
    return " ".join(cli)


for trial in range(N_RUNS):
    for dataset in DATASETS:
        print(f"# {dataset}")
        for method in METHODS:
            print(run_string(trial, dataset, method))
    print()
