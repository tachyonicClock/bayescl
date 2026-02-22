DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]
METHODS = [
    "01_linear",
    "02_lora",
    "03_ball",
    # "04_replay",
    # "05_gdumb",
    # "06_der",
    "07_joint",
    "08_rwalk",
    # "09_l2p",
    "10_ewc",
    "11_tball",
    # "12_rball",
    # "13_rtball",
]
N_RUNS = 5

def run_string(trial: int, dataset: str, method: str):
    label = f"{dataset[:5]}_{method[3 : 3 + 5]}"
    cli = [
        f"ts -G 1 -L {label:<11}",
        "notirun.sh",
        "python main.py",
        f"-c configs/{dataset}/{method}.yaml".rjust(35),
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