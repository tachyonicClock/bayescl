from itertools import product
N=5
DATASETS = [
    "cifar100",
    "imagenetr",
]
METHODS = [
    "01_linear",
    "02_lora",
    "03_blob",
    "04_clora",
    "05_inflora",
]

for dataset, method in product(DATASETS, METHODS):
    print(
        f"ts -G 1 -m -L {method:<10} python main.py --hpsearch --configs configs/{dataset}/{method}.yaml"
    )


for dataset, method in product(DATASETS, METHODS):
    print(
        f"ts -G 1 -m -L {method:<10} python main.py --repeat={N} --configs configs/{dataset}/{method}.yaml"
    )
