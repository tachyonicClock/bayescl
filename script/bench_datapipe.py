"""Micro-benchmark the HF-backed data pipeline for ImageNet-R and CORe50.

Measures raw ``__getitem__`` throughput and ``DataLoader`` throughput at a few
worker counts so pipeline bottlenecks are visible.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from bayescl.data.benchmark import get_transforms
from bayescl.data.datasets import CORe50Dataset, ImageNetR, datasets_path


def time_raw(ds, n):
    idx = torch.randperm(len(ds))[:n].tolist()
    t = time.time()
    for i in idx:
        ds[i]
    dt = time.time() - t
    print(f"  raw __getitem__: {n / dt:8.1f} img/s ({dt:.2f}s for {n})")


def time_loader(ds, batch_size, workers, batches):
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        persistent_workers=workers > 0,
        pin_memory=True,
        prefetch_factor=4 if workers > 0 else None,
    )
    it = iter(dl)
    next(it)  # warm up workers / first batch
    t = time.time()
    seen = 0
    for _ in range(batches):
        x, *_ = next(it)
        seen += x.shape[0]
    dt = time.time() - t
    print(
        f"  DataLoader workers={workers:2d}: {seen / dt:8.1f} img/s ({dt:.2f}s for {seen})"
    )
    del it, dl


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["imagenetr", "core50", "cifar100"], default="imagenetr"
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--raw-n", type=int, default=512)
    args = p.parse_args()

    scenario = {
        "imagenetr": "ImageNetR",
        "core50": "CORe50",
        "cifar100": "CIFAR100",
    }[args.dataset]
    train_tf, eval_tf = get_transforms(standardize=True, dataset=scenario)

    t = time.time()
    if args.dataset == "imagenetr":
        ds = ImageNetR(transform=train_tf)
    elif args.dataset == "core50":
        ds = CORe50Dataset(split="train&valid", transform=train_tf)
    else:
        ds = CIFAR100(datasets_path(), train=True, transform=train_tf)
    print(f"construct {args.dataset}: {time.time() - t:.2f}s, len={len(ds)}")

    time_raw(ds, args.raw_n)
    for w in (0, 4, 8):
        time_loader(ds, args.batch_size, w, args.batches)


if __name__ == "__main__":
    main()
