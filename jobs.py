from itertools import product

N = 5
DATASETS = [
    "cifar100",
    "imagenetr",
    "domainnet",
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
]

# for dataset, method in product(DATASETS, METHODS):
for method, dataset in product(METHODS, DATASETS):
    label = f"{dataset[:5]}_{method[3 : 5 + 3]}"
    print(
        f"ts -G 1 -L {label:<11} python main.py --hpsearch -c configs/{dataset}/{method}.yaml"
    )
