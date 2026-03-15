from claiutil.cmap import xgfs_normal12

N_RUNS = 5

OFFLINE = {"joint"}


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
}

METHOD_TO_MARKER = {
    None: ".",
}

METHOD_TO_LINESTYLE = {
    None: "-",
}

COLORS = {
    "tball": xgfs_normal12(0),
    "linear": xgfs_normal12(1),
    "joint": xgfs_normal12(2),
    "rwalk": xgfs_normal12(3),
    "ball": xgfs_normal12(4),
    "ewc": xgfs_normal12(5),
    "mas": xgfs_normal12(10),
    "si": xgfs_normal12(9),
    "lora": xgfs_normal12(6),
    "clora": xgfs_normal12(7),
    "sdlora": xgfs_normal12(8),
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

BACKBONES = {
    "joint": "DinoV2-small",
    "codap": "ViT-B/16",
    "dualprompt": "ViT-B/16",
    "l2p": "ViT-B/16",
    "replay": "DinoV2-small",
    "gdumb": "DinoV2-small",
    "der": "DinoV2-small",
    "lora": "DinoV2-small",
    "linear": "DinoV2-small",
    "rwalk": "DinoV2-small",
    "ball": "DinoV2-small",
}

REPLAY_METHODS = {"replay", "gdumb", "der"}

PLOT_METHODS = [
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
]


OFFLINE_COMPATIBLE_METRICS = {
    "accuracy_final",
    "ece_final",
    "ace_final",
    "sce_final",
    "brier_final",
    "asce_final",
}


def get_color(method_key: str) -> str:
    """Get the color for a given method."""
    return COLORS.get(method_key, "red")


def get_label(method_key: str) -> str:
    """Get the label for a given method."""
    return METHOD_TO_LABEL.get(method_key, method_key)


def get_marker(method_key: str) -> str:
    """Get the marker for a given method."""
    return METHOD_TO_MARKER.get(method_key, METHOD_TO_MARKER[None])


def get_linestyle(method_key: str) -> str:
    """Get the line style for a given method."""
    return METHOD_TO_LINESTYLE.get(method_key, METHOD_TO_LINESTYLE[None])
