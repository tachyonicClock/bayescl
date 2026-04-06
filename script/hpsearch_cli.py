DATASETS = [
    # "cifar100",
    # "core50",
    # "imagenetr",
    "cifar100-sensitivity"
]

METHODS = [
    # "rwalk",
    # "ewc",
    # "lora",
    # "clora",
    # "sdlora",
    # "inflora",
    "ball",
    "tball",
    "tball-mnd",
]

# 5 CHARS MAX
ABBREVIATIONS = {
    "cifar100": "C100",
    "core50": "C50",
    "imagenetr": "INR",
    "rwalk": "RWLK",
    "ewc": "EWC",
    "lora": "LORA",
    "clora": "CLORA",
    "sdlora": "SDLORA",
    "inflora": "INFLO",
    "ball": "BALL",
    "tball": "TBALL",
    "tball-mnd": "TBMND",
    "cifar100-sensitivity": "C100S",
}


def run_string(dataset, method):
    label = f"{ABBREVIATIONS[dataset]}.{ABBREVIATIONS[method]}"
    cli = [
        f"ts -G 1 -L {label:<11}",
        "notirun.sh",
        "python main.py",
        f"-c configs/{dataset}/{method}.jsonnet".rjust(40),
        "hpsearch sens",
    ]
    return " ".join(cli)


for dataset in DATASETS:
    for method in METHODS:
        print(run_string(dataset, method))
    print()
