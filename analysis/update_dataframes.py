import pickle
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import numpy as np
import pandas as pd
import torch
from calibration import (
    calibration_curve as compute_calibration_curve,
)
from calibration import (
    expected_calibration_error,
)
from torch import Tensor
from torchmetrics.utilities.compute import normalize_logits_if_needed
import os

DATASET_NAME_MAP = {
    # "ImageNet_R": "imagenetr",
    "core50": "core50",
    "cifar100": "cifar100",
    "domainnet": "domainnet",
}

DATASETS = [
    "cifar100",
    "domainnet",
    "imagenetr",
]


@dataclass
class SummaryRecord:
    dataset: str
    method: str
    run_id: str
    n_tasks: int
    accuracy_all_avg: float
    accuracy_final: float
    accuracy_seen_avg: float
    ace_all_avg: float
    ace_final: float
    ace_seen_avg: float
    backward_transfer: float
    brier_all_avg: float
    brier_final: float
    ece_final: float
    ece_seen_avg: float
    ece_all_avg: float
    forward_transfer: float
    sce_all_avg: float
    sce_final: float
    sce_seen_avg: float
    asce_final: float


@dataclass
class TimeSeriesRecord:
    dataset: str
    method: str
    run_id: str
    task: int
    accuracy_all: float
    accuracy_seen: float
    ace_all: float
    ace_seen: float
    brier_all: float
    ece_all: float
    ece_seen: float
    sce_all: float
    sce_seen: float


@dataclass
class CalibrationRecord:
    dataset: str
    method: str
    run_id: str
    bin: int
    bin_probability: float
    bin_frequency: float
    bin_weight: float


def extract_inflora(
    filename: Path | str,
) -> Generator[Tuple[str, str, str, Dict[str, Any]], None, None]:
    filename = Path(filename)
    with tarfile.open(filename, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("record.pkl"):
                f = tar.extractfile(member)
                if f is not None:
                    parts = member.name.split("/")
                    dataset = parts[2]
                    method = parts[4]
                    run_id = parts[6]
                    data = pickle.load(f)
                    yield dataset, method, run_id, data


def extract_bayescl(
    filename: Path | str,
) -> Generator[Tuple[str, str, str, Dict[str, Any]], None, None]:
    filename = Path(filename)
    with tarfile.open(filename, mode="r") as tar:
        for member in tar.getmembers():
            if member.name.endswith("metrics.pkl"):
                f_metrics = tar.extractfile(member)
                f_raw_data = tar.extractfile(
                    (Path(member.name).parent / "raw_data.pkl").as_posix()
                )
                if f_metrics is None or f_raw_data is None:
                    continue
                metrics = pickle.load(f_metrics)
                raw_data = pickle.load(f_raw_data)
                metrics.update(raw_data)

                parts = member.name.split("/")
                dataset = parts[3]
                method = parts[4]
                run_id = parts[5]
                yield dataset, method, run_id, metrics


def from_logs(filename: Path) -> Generator[Tuple[str, str, str, Dict[str, Any]], None, None]:
    # log/test/cifar100/ball/00/metrics.pkl
    # log/test/cifar100/ball/00/raw_data.pkl
    for root, _, files in os.walk(filename):
        if "metrics.pkl" in files and "raw_data.pkl" in files:
            with open(os.path.join(root, "metrics.pkl"), "rb") as f_metrics, open(
                os.path.join(root, "raw_data.pkl"), "rb"
            ) as f_raw_data:
                metrics = pickle.load(f_metrics)
                raw_data = pickle.load(f_raw_data)
                metrics.update(raw_data)

                parts = root.split(os.sep)
                dataset = parts[2]
                method = parts[3]
                run_id = parts[4]
                yield dataset, method, run_id, metrics

def get_final_proba_and_targets(data) -> Tuple[Tensor, Tensor]:
    n_tasks = max(train_id for (train_id, _) in data["y_logit"].keys()) + 1

    y_logits = torch.concat(
        [
            torch.from_numpy(data["y_logit"][(n_tasks - 1, i)]).float()
            for i in range(n_tasks)
        ],
        dim=0,
    )  # type: ignore
    y_true = torch.concat(
        [
            torch.from_numpy(data["y_true"][(n_tasks - 1, i)]).float()
            for i in range(n_tasks)
        ],
        dim=0,
    )  # type: ignore
    proba = normalize_logits_if_needed(y_logits, "softmax")
    return proba, y_true


def to_summary_record(
    dataset: str, method: str, run_id: str, data: Dict[str, Any]
) -> SummaryRecord:
    proba, targets = get_final_proba_and_targets(data)
    bin_probability, bin_frequency, bin_weights = compute_calibration_curve(
        proba.numpy(),
        targets.numpy(),
        num_bins=15,
        equal_size_bins=False,
        top_class_only=False,
    )
    asce_final = float(
        expected_calibration_error(bin_probability, bin_frequency, bin_weights)
    )

    return SummaryRecord(
        dataset=DATASET_NAME_MAP.get(dataset, dataset),
        method=method,
        run_id=run_id,
        n_tasks=data["n_tasks"],
        accuracy_all_avg=data["accuracy_all_avg"],
        accuracy_final=data["accuracy_final"],
        accuracy_seen_avg=data["accuracy_seen_avg"],
        ace_all_avg=data["ace_all_avg"],
        ace_final=data["ace_final"],
        ace_seen_avg=data["ace_seen_avg"],
        backward_transfer=data["backward_transfer"],
        brier_all_avg=data["brier_all_avg"],
        brier_final=data["brier_final"],
        ece_final=data["ece_final"],
        ece_seen_avg=data["ece_seen_avg"],
        forward_transfer=data["forward_transfer"],
        sce_all_avg=data["sce_all_avg"],
        sce_final=data["sce_final"],
        sce_seen_avg=data["sce_seen_avg"],
        ece_all_avg=data["ece_all_avg"],
        asce_final=asce_final,
    )


def calibration_curve(
    data, equal_size_bins: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba, targets = get_final_proba_and_targets(data)
    bin_probability, bin_frequency, bin_weights = compute_calibration_curve(
        proba.numpy(), targets.numpy(), num_bins=15, equal_size_bins=equal_size_bins
    )
    return (
        np.nan_to_num(bin_probability, nan=0),
        np.nan_to_num(bin_frequency, nan=0),
        np.nan_to_num(bin_weights, nan=0),
    )


def to_calibration_record(
    dataset: str, method: str, run_id: str, data: Dict[str, Any]
) -> Generator[CalibrationRecord, None, None]:
    bin_probability, bin_frequency, bin_weights = calibration_curve(data)
    n_bins = len(bin_probability)
    for i in range(n_bins):
        yield CalibrationRecord(
            dataset=DATASET_NAME_MAP.get(dataset, dataset),
            method=method,
            run_id=run_id,
            bin=i,
            bin_probability=bin_probability[i],
            bin_frequency=bin_frequency[i],
            bin_weight=bin_weights[i],
        )


def to_timeseries_record(
    dataset: str, method: str, run_id: str, data: Dict[str, Any]
) -> Generator[TimeSeriesRecord, None, None]:
    n_tasks = data["n_tasks"]
    for task in range(n_tasks):
        yield TimeSeriesRecord(
            dataset=DATASET_NAME_MAP.get(dataset, dataset),
            method=method,
            run_id=run_id,
            task=task,
            accuracy_all=data["accuracy_all"][task],
            accuracy_seen=data["accuracy_seen"][task],
            ace_all=data["ace_all"][task],
            ace_seen=data["ace_seen"][task],
            brier_all=data["brier_all"][task],
            ece_all=data["ece_all"][task],
            ece_seen=data["ece_seen"][task],
            sce_all=data["sce_all"][task],
            sce_seen=data["sce_seen"][task],
        )


def dataclass_to_df(records: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([asdict(record) for record in records])


summary_records = []
time_series_records = []
calibration_records = []

# BayesCL extraction and transformations
for dataset, method, run_id, data in from_logs("log/test"):
    print(f"Processing {dataset}/{method}/{run_id}")
    summary_records.append(to_summary_record(dataset, method, run_id, data))
    for record in to_timeseries_record(dataset, method, run_id, data):
        time_series_records.append(record)
    for record in to_calibration_record(dataset, method, run_id, data):
        calibration_records.append(record)

dataclass_to_df(summary_records).to_parquet("analysis/dataframe/summary.parquet")
dataclass_to_df(time_series_records).to_parquet("analysis/dataframe/time_series.parquet")
dataclass_to_df(calibration_records).to_parquet("analysis/dataframe/calibration.parquet")
