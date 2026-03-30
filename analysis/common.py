from typing import Any

N_RUNS = 5

OFFLINE = {"joint"}

METHODS = [
    # "joint",
    "lora",
    "ewc",
    "rwalk",
    "clora",
    "sdlora",
    "inflora",
    "ball",
    "tball-mnd",
    "tball",
]

DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]

#: Maps method names to their display labels.
METHOD_TO_LABEL = {
    "rwalk": "RWalk",
    "ewc": "EWC",
    "lora": "LoRA",
    "clora": "C-LoRA",
    "sdlora": "SD-LoRA",
    "inflora": "InfLoRA",
    "ball": "BALL",
    "tball-mnd": r"TBALL\textsubscript{MND}",
    "tball": r"TBALL",
}

MARKERS = [".", "x"]
LINE_STYLES = ["-", "--"]

CMAP_COLOURS = [
    (1.0, 0.62301366, 0.77227656),
    (0.79695073, 0.01601229, 0.66400962),
    (0.35504226, 0.14258962, 0.23456003),
    (0.81050133, 0.08462097, 0.07466861),
    (1.0, 0.57340844, 0.03354555),
    (0.79638876, 0.91057152, 0.17576824),
    (0.14887713, 0.57919728, 0.12758598),
    (0.08817248, 0.80514856, 0.89713061),
    (0.56116415, 0.40395853, 0.97883405),
    (0.00795099, 0.1623914, 0.69189112),
]
# CMAP_COLOURS = sns.color_palette("Paired")[0:10]

_STYLE = {
    "lora": {
        "color": CMAP_COLOURS[0],
        "marker": "o",
        "linestyle": "-",
    },
    "ewc": {
        "color": CMAP_COLOURS[1],
        "marker": "s",
        "linestyle": ":",
    },
    "rwalk": {
        "color": CMAP_COLOURS[4],
        "marker": "D",
        "linestyle": "-",
    },
    "clora": {
        "color": CMAP_COLOURS[5],
        "marker": "^",
        "linestyle": ":",
    },
    "sdlora": {
        "color": CMAP_COLOURS[6],
        "marker": "v",
        "linestyle": "-",
    },
    "inflora": {
        "color": CMAP_COLOURS[7],
        "marker": "o",
        "linestyle": ":",
    },
    "tball": {
        "color": CMAP_COLOURS[3],
        "marker": "s",
        "linestyle": ":",
    },
    "ball": {
        "color": CMAP_COLOURS[9],
        "marker": "D",
        "linestyle": "-",
    },
    "tball-mnd": {
        "color": CMAP_COLOURS[8],
        "marker": "p",
        "linestyle": "--",
    },
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
    "core50": "iCORe50/10",
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
    "ace_seen_avg": r"Avg. ACE $\downarrow$",
    "score": r"Score $\uparrow$",
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
