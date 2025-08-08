for dataset in [
    # "core50",
    "cifar100",
    # "imagenetr"
]:
    for method in ["01_linear", "02_lora", "03_blob", "04_clora", "05_inflora"]:
        print(
            f"ts -G 1 -m -L {method:<10} python main.py --hpsearch --configs configs/{dataset}/{method}.yaml"
        )
