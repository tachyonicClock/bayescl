DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]

METHODS = [
    "ball",
    "ewc",
    "joint",
    "lora",
    "rwalk",
    "sdlora",
    "tball",
]


def run_string(dataset, method):
    label = f"{dataset[:5]}_{method[:5]}"
    cli = [
        f"ts -G 1 -L {label:<11}",
        "notirun.sh",
        "python main.py",
        f"-c configs/{dataset}/{method}.yaml".rjust(35),
        "hpsearch hp",
    ]
    return " ".join(cli)


for method in METHODS:
    for dataset in DATASETS:
        print(run_string(dataset, method))
    print()
