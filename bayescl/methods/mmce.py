from torch import Tensor
import torch
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin
from typing import Any
from bayescl.config import MMCEConfig
from torch.utils.tensorboard import SummaryWriter

def calibration_mmce_w_loss(logits: Tensor, correct_labels: Tensor) -> Tensor:
    """Function to compute the MMCE_w loss in PyTorch.

    This function was translated from TensorFlow (https://github.com/aviralkumar2907/MMCE/blob/master/20ng_mmce.py#L151)
    using Gemini Pro. I have verified that the outputs match for a variety of random inputs.

    :param logits: Tensor of shape (batch_size, num_classes) representing the model logits.
    :param correct_labels: Tensor of shape (batch_size,) representing the correct class labels.
    :return: Scalar Tensor representing the MMCE_w loss.
    """
    # 1. Get probabilities and predictions
    predicted_probs = F.softmax(logits, dim=1)
    predicted_probs_max, predicted_labels = torch.max(predicted_probs, dim=1)

    # 2. Create mask for correct predictions
    correct_mask = torch.eq(predicted_labels, correct_labels)
    
    # 3. Separate probabilities into correct and incorrect sets
    # Note: The original TF code uses top_k with masking to extract these. 
    # In PyTorch, boolean indexing is more direct and equivalent.
    correct_prob = predicted_probs_max[correct_mask]
    incorrect_prob = predicted_probs_max[~correct_mask]

    # Get counts (m and n in the original paper/code)
    m = correct_prob.size(0)
    n = incorrect_prob.size(0)

    # 4. Handle Edge Cases
    # The original code forces the result to 0 if either m or n is 0 
    # (via cond_k * cond_k_p logic). We handle this explicitly here.
    if m == 0 or n == 0:
        return torch.tensor(0.0, device=logits.device)

    # 5. Define the Kernel Function
    # The original code uses a Laplacian kernel with sigma = 0.2 (2 * sigma = 0.4)
    # PyTorch broadcasting handles the pairwise matrix generation automatically.
    def compute_kernel_matrix(vec1, vec2):
        # vec1: shape (A,), vec2: shape (B,)
        # unsqueeze allows broadcasting to shape (A, B)
        # abs(vec1[i] - vec2[j])
        diff = torch.abs(vec1.unsqueeze(1) - vec2.unsqueeze(0))
        return torch.exp(-1.0 * diff / 0.4)

    # 6. Compute Kernels
    correct_kernel = compute_kernel_matrix(correct_prob, correct_prob)
    incorrect_kernel = compute_kernel_matrix(incorrect_prob, incorrect_prob)
    correct_incorrect_kernel = compute_kernel_matrix(correct_prob, incorrect_prob)

    # 7. Compute Sampling Weights (Outer Products)
    # Original: matmul((1-p), (1-p).T)
    w_correct = 1.0 - correct_prob
    sampling_weights_correct = torch.matmul(
        w_correct.unsqueeze(1), w_correct.unsqueeze(0)
    )

    # Original: matmul(p, p.T)
    w_incorrect = incorrect_prob
    sampling_weights_incorrect = torch.matmul(
        w_incorrect.unsqueeze(1), w_incorrect.unsqueeze(0)
    )

    # Original: matmul((1-p_c), p_i.T)
    sampling_correct_incorrect = torch.matmul(
        w_correct.unsqueeze(1), w_incorrect.unsqueeze(0)
    )

    # 8. Compute Output Values (Means of kernel * weights)
    correct_correct_vals = (correct_kernel * sampling_weights_correct).mean()
    incorrect_incorrect_vals = (incorrect_kernel * sampling_weights_incorrect).mean()
    correct_incorrect_vals = (correct_incorrect_kernel * sampling_correct_incorrect).mean()

    # 9. Compute MMD Error
    # Note: We add epsilon to denominators to match the stability epsilon in TF code
    mmd_error = (1.0 / (m * m + 1e-5)) * torch.sum(correct_correct_vals)
    mmd_error += (1.0 / (n * n + 1e-5)) * torch.sum(incorrect_incorrect_vals)
    mmd_error -= (2.0 / (m * n + 1e-5)) * torch.sum(correct_incorrect_vals)

    # 10. Return final loss
    # sqrt(relu(mmd)) to ensure non-negative before sqrt
    return torch.sqrt(torch.relu(mmd_error + 1e-10))


class MMCEPlugin(SupervisedPlugin):
    """Plugin to compute MMCE loss during training."""

    def __init__(self, config: MMCEConfig, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.writer = writer

    def after_forward(self, strategy: Any, *args, **kwargs) -> Any:
        logits = strategy.mb_output
        targets = strategy.mb_y
        mmce_loss = calibration_mmce_w_loss(logits, targets).item()
        strategy.loss += self.config.weight * mmce_loss

        # Log MMCE loss
        step = strategy.clock.train_iterations
        self.writer.add_scalar("mmce/loss", mmce_loss, step)
