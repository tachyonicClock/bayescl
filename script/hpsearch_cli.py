DATASETS = [
    "cifar100",
    "domainnet",
    "imagenetr",
]

METHODS = [
    "01_linear",
    "02_lora",
    "03_ball",
    "04_replay",
    "05_gdumb",
    "06_der",
    "07_joint",
    "08_rwalk",
    "09_l2p",
    "10_ewc",
    "11_tball",
    "12_rball",
    "13_rtball",
]


def run_string(dataset, method):
    label = f"{dataset[:5]}_{method[3 : 3 + 5]}"
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
