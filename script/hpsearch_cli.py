DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]

METHODS = [
    "rwalk",
    "ewc",
    "lora",
    "clora",
    "sdlora",
    "inflora",
    "ball",
    "tball",
    "tball-mnd",
]


def run_string(dataset, method):
    label = f"{dataset[:5]}_{method[:5]}"
    cli = [
        f"ts -G 1 -L {label:<11}",
        "notirun.sh",
        "python main.py",
        f"-c configs/{dataset}/{method}.jsonnet".rjust(40),
        "hpsearch hp",
    ]
    return " ".join(cli)


for dataset in DATASETS:
    for method in METHODS:
        print(run_string(dataset, method))
    print()
