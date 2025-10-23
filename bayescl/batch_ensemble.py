import math

import torch
from torch import Tensor, nn

from bayescl.vbnn import VariationalParameter, VBNNConfig


def ensemble_predict(input: Tensor, module: nn.Module, ensemble_size: int) -> Tensor:
    """_summary_

    :param input: tensor of shape (batch_size, in_features)
    :param module: a module that accepts input of shape (ensemble_size*batch_size, *)
        and outputs tensor of shape (ensemble_size*batch_size, num_classes)
    :param ensemble_size: size of the ensemble
    :return: tensor of shape (ensemble_size, batch_size, num_classes)
    """
    es = ensemble_size  # ensemble size
    bs = input.size(0)  # batch size
    ndim = input.dim()

    # Repeat the batches such that the shape is (es*bs, *) where * is the remaining
    # dimensions
    inputs = torch.tile(input, (es, *(1,) * (ndim - 1)))
    outputs = module(inputs)  # type: ignore
    return outputs.view(es, bs, -1)


def _reset_parameters(weight: Tensor, bias: Tensor | None, ensemble_size: int) -> None:
    for k in range(ensemble_size):
        nn.init.kaiming_uniform_(weight[k], a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight[k])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias[k], -bound, bound)


def batch_ensemble_linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """Functional implementation of the BatchEnsemble linear layer.

    :param input: Input tensor of shape ``(*, in_features)`` where ``*`` represents any number
        of leading dimensions. The * leading dimensions must multiply to a value divisible by the
        ensemble size.
    :param weight: Weight tensor of shape ``(ensemble_size, out_features, in_features)``.
    :param bias: Optional bias tensor of shape ``(ensemble_size, out_features)``
    :return: Output tensor of shape ``(*, out_features)``
    """
    ensemble_size = weight.size(0)
    out_features = weight.size(1)
    in_features = weight.size(2)
    batch_dims = input.shape[:-1]
    if math.prod(batch_dims) % ensemble_size != 0:
        raise ValueError("The batch dimensions must be divisible by the ensemble size.")

    # Reshape input to (ensemble_size, examples_per_model, in_features)
    input_reshaped = input.view(ensemble_size, -1, in_features)

    # Perform batched matrix multiplication.
    output = torch.bmm(input_reshaped, weight.transpose(1, 2))

    # Add bias if present
    if bias is not None:
        output += bias.unsqueeze(1)

    # Reshape back to (*, out_features)
    return output.view(*batch_dims, out_features)


class BatchEnsembleLinear(nn.Module):
    """A linear layer that implements the BatchEnsemble method for efficient
    ensembling.

    Wen, Yeming, Dustin Tran, and Jimmy Ba. “BatchEnsemble: An Alternative Approach to
    Efficient Ensemble and Lifelong Learning.” 8th International Conference on Learning
    Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020,
    OpenReview.net, 2020. https://openreview.net/forum?id=Sklf1yrYDr.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # Define parameters
        self.weight = nn.Parameter(
            torch.zeros(ensemble_size, out_features, in_features)
        )
        """The learnable weights of the module of shape ``(ensemble_size, out_features,
        in_features)``"""
        self.bias = (
            nn.Parameter(torch.zeros(ensemble_size, out_features)) if bias else None
        )
        """Optional learnable bias of the module of shape ``(ensemble_size, out_features)``"""
        _reset_parameters(self.weight, self.bias, ensemble_size)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the BatchEnsembleLinear layer.

        :param input: Input tensor of shape ``(ensemble_size*examples_per_model,
            in_features)`` or equivalently ``(batch_size, in_features)`` where
            ``batch_size = ensemble_size * examples_per_model``.
        :return: Output tensor of shape ``(batch_size, out_features)``
        """
        return batch_ensemble_linear(input, self.weight, self.bias)


class BayesianBatchEnsembleLinear(nn.Module):
    """A Bayesian linear layer that implements the BatchEnsemble method for efficient
    ensembling using Bayesian weights.

    Dusenberry, Michael W., Ghassen Jerfel, Yeming Wen, et al. “Efficient and Scalable
    Bayesian Neural Nets with Rank-1 Factors.” arXiv:2005.07186. Preprint, arXiv, August
    14, 2020. https://doi.org/10.48550/arXiv.2005.07186.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
        config: VBNNConfig = VBNNConfig(),
    ):
        super().__init__()
        # Store arguments
        self._in_features = in_features
        self._out_features = out_features
        self._ensemble_size = ensemble_size

        # Define parameters
        self.weight = VariationalParameter(
            shape=(ensemble_size, out_features, in_features), config=config
        )
        """The learnable weights of the module of shape ``(ensemble_size, out_features,
            in_features)``"""
        self.bias = None
        """Optional learnable bias of the module of shape ``(ensemble_size, out_features)
        """
        if bias:
            self.bias = VariationalParameter(
                shape=(ensemble_size, out_features), config=config
            )

        _reset_parameters(
            self.weight.mu, self.bias.mu if self.bias else None, ensemble_size
        )

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the BayesianBatchEnsembleLinear layer.

        :param input: Input tensor of shape ``(ensemble_size*examples_per_model,
            in_features)`` or equivalently ``(batch_size, in_features)`` where
            ``batch_size = ensemble_size * examples_per_model``.
        :return: Output tensor of shape ``(batch_size, out_features)``
        """
        weight = self.weight.forward()
        bias = self.bias.forward() if self.bias is not None else None
        return batch_ensemble_linear(input, weight, bias)
