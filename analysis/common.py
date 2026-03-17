from typing import Any

import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

N_RUNS = 5

OFFLINE = {"joint"}

METHODS = [
    "joint",
    "rwalk",
    "ewc",
    "mas",
    "si",
    "lora",
    "clora",
    "sdlora",
    "ball",
    "tball",
    "inflora",
]

DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]

#: Maps method names to their display labels.
METHOD_TO_LABEL = {
    "joint": "Offline LoRA",
    # "linear": "Linear",
    "rwalk": "RWalk",
    "ewc": "EWC",
    "mas": "MAS",
    "si": "SI",
    "lora": "LoRA",
    "clora": "C-LoRA",
    "sdlora": "SD-LoRA",
    "ball": "BALL",
    "tball": "TBALL",
    "inflora": "InfLoRA",
}

MARKERS = [".", "x"]
LINE_STYLES = ["-", "--"]

CMAP_COLOURS = get_cmap("tab20").colors
# CMAP_COLOURS = sns.color_palette("Paired")[0:10]
offset = 0

_STYLE = {
    method: {
        "color": mcolors.to_hex(CMAP_COLOURS[i % len(CMAP_COLOURS)]),
        "marker": MARKERS[i % len(MARKERS)],
        "linestyle": LINE_STYLES[i % len(LINE_STYLES)],
    }
    for i, method in enumerate(METHODS, start=offset)
}
_STYLE["None"] = {
    "color": "black",
    "marker": "v",
    "linestyle": "-",
}

#: Maps dataset names to their display labels.
DATASET_TO_LABEL = {
    "cifar100": "iCIFAR100/10",
    # "domainnet": "iDomainNet345/5",
    "core50": "iCore50/10",
    "imagenetr": "iImageNet-R200/10",
}

METRIC_TO_LABEL = {
    "accuracy_seen": r"Acc. $\uparrow$",
    "accuracy_seen_avg": r"Avg. Acc. $\uparrow$",
    "accuracy_final": r"Acc. $\uparrow$",
    "backward_transfer": r"BWT $\uparrow$",
    "ece_seen_avg": r"Avg. ECE $\downarrow$",
    "ece_all_avg": r"Avg. All ECE $\downarrow$",
    "ece_final": r"ECE $\downarrow$",
    "ace_final": r"ACE $\downarrow$",
    "sce_final": r"SCE $\downarrow$",
    "brier_final": r"Brier $\downarrow$",
    "asce_final": r"ASCE $\downarrow$",
}

REPLAY_METHODS = {"replay", "gdumb", "der"}


OFFLINE_COMPATIBLE_METRICS = {
    "accuracy_final",
    "ece_final",
    "ace_final",
    "sce_final",
    "brier_final",
    "asce_final",
}


def get_color(method_key: str) -> Any:
    """Get the color for a given method."""
    return _STYLE.get(method_key, _STYLE["None"])["color"]


def get_method_label(method_key: str) -> str:
    """Get the label for a given method."""
    return METHOD_TO_LABEL.get(method_key, method_key)


def get_marker(method_key: str) -> Any:
    """Get the marker for a given method."""
    return _STYLE.get(method_key, _STYLE["None"])["marker"]


def get_linestyle(method_key: str) -> Any:
    """Get the line style for a given method."""
    return _STYLE.get(method_key, _STYLE["None"])["linestyle"]


def get_dataset_label(dataset_key: str) -> str:
    """Get the label for a given dataset."""
    return DATASET_TO_LABEL.get(dataset_key, dataset_key)
