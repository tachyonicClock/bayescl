for dataset in [
    # "core50",
    "cifar100",
    # "imagenetr"
]:
    for method in ["01_linear", "02_lora", "03_blob", "04_clora"]:
        print(f"python main.py --hpsearch 1 --configs configs/{dataset}/{method}.yaml")
