import matplotlib.figure
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_confusion_matrix(counts: Tensor, title: str = "") -> matplotlib.figure.Figure:
    """Row-normalized (per true class) confusion matrix heatmap.

    ``counts[i, j]`` is the number of samples with true label ``i`` predicted
    as ``j`` (see ``ContinualLearningEvaluator.checkpoint_confusion_counts``).
    Rows with no samples are left at 0 rather than dividing by zero.
    """
    counts = counts.double()
    row_totals = counts.sum(dim=1, keepdim=True)
    normalized = torch.where(
        row_totals > 0, counts / row_totals.clamp(min=1), torch.zeros_like(counts)
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(normalized.numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
