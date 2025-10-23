import pytest
import torch
import torch.nn as nn

from bayescl.batch_ensemble import BatchEnsembleLinear, ensemble_predict


def test_ensemble_predict():
    bs = 6  # batch size
    es = 4  # ensemble size
    in_features = 1

    input = torch.arange(bs * in_features).view(bs, in_features).float()
    module = nn.Identity()
    outputs = ensemble_predict(input, module, es)

    assert outputs.shape == (es, bs, in_features)
    assert torch.allclose(outputs.mean(0), input)


def test_like_linear():
    """When ensemble_size=1, BatchEnsembleLinear should behave exactly like nn.Linear."""
    in_features = 10
    out_features = 5
    ensemble_size = 1
    batch_size = 8

    torch.manual_seed(0)
    layer_be = BatchEnsembleLinear(in_features, out_features, ensemble_size)

    torch.manual_seed(0)
    layer_linear = nn.Linear(in_features, out_features)

    # Ensure weights and biases are the same
    assert torch.allclose(layer_be.weight[0], layer_linear.weight)
    assert layer_be.bias is not None and torch.allclose(
        layer_be.bias[0], layer_linear.bias
    )

    # Create random input
    input = torch.randn(batch_size, in_features)
    output_be = layer_be(input)
    output_linear = layer_linear(input)
    assert torch.allclose(output_be, output_linear)

    # Verify gradients
    output_be.sum().backward()
    output_linear.sum().backward()
    assert layer_be.weight.grad is not None and layer_linear.weight.grad is not None
    assert torch.allclose(layer_be.weight.grad[0], layer_linear.weight.grad)
    assert layer_be.bias is not None and layer_linear.bias is not None
    assert layer_be.bias.grad is not None and layer_linear.bias.grad is not None
    assert torch.allclose(layer_be.bias.grad[0], layer_linear.bias.grad)

    # Linear can handle extra dimensions in input
    input = torch.randn(2, 3, in_features)
    output_be = layer_be(input)
    output_linear = layer_linear(input)
    assert torch.allclose(output_be, output_linear)


@pytest.mark.xfail(raises=ValueError)
def test_incompatible_batch_size():
    BatchEnsembleLinear(10, 5, ensemble_size=4).forward(torch.randn(10, 10))


def test_batch_ensemble():
    """Test BatchEnsembleLinear with ensemble_size > 1."""
    in_features = 10
    out_features = 5
    ensemble_size = 3
    examples_per_model = 4
    batch_size = ensemble_size * examples_per_model

    layer_be = BatchEnsembleLinear(in_features, out_features, ensemble_size)

    # Create random input
    input = torch.randn(batch_size, in_features)
    output_be = layer_be(input)

    # Verify output for each ensemble member comparing to nn.Linear
    for k in range(ensemble_size):
        x = input[k * examples_per_model : (k + 1) * examples_per_model]
        expected = torch.nn.functional.linear(
            x,
            layer_be.weight[k],
            layer_be.bias[k] if layer_be.bias is not None else None,
        )
        assert torch.allclose(
            output_be[k * examples_per_model : (k + 1) * examples_per_model], expected
        )
