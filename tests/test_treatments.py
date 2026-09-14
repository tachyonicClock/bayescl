"""Parameterized constructor / parameter-count / forward-pass tests for every
registered treatment (arm).

Exercises each arm's ``_build_peft`` against a tiny stand-in model that mirrors
the module names (``vit.encoder.layer.N....`` / ``classifier``) of the real
production backbone (see :class:`bayescl.config.Backbone`), so the tests catch
mismatches between an arm's adapter factory and the actual (Linear-based)
adapter_filter/head_module, not just isolated unit behaviour.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import bayescl.arms  # noqa: F401  (import side effect: populates ARMS)
from bayescl.config import Backbone
from bayescl.model import HuggingFaceImageClassifierAdapter
from bayescl.treatments._registry import ARMS, ArmBase, get_arm
from bayescl.treatments.lora_ensemble._module import EnsembleModule

HIDDEN = 8
INTERMEDIATE = 16
N_LAYERS = 2
NUM_CLASSES = 4
BATCH = 3
N_TASKS = 3

BACKBONE = Backbone()

# Arms whose adapters are populated lazily (by a plugin, during training)
# rather than eagerly in ``_build_peft``, so no non-head parameters are
# trainable immediately after construction.
LAZY_ADAPTER_ARMS = {"inflora"}


class _SelfAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)


class _SelfOutput(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class _Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attention = _SelfAttention(hidden)
        self.output = _SelfOutput(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.attention.query(x)
        k = self.attention.key(x)
        v = self.attention.value(x)
        return self.output(q + k + v)


class _Intermediate(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.dense = nn.Linear(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.dense(x))


class _Output(nn.Module):
    def __init__(self, intermediate: int, hidden: int):
        super().__init__()
        self.dense = nn.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class _EncoderLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.attention = _Attention(hidden)
        self.intermediate = _Intermediate(hidden, intermediate)
        self.output = _Output(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.output(self.intermediate(x))


class _TinyViT(nn.Module):
    """Mirrors ``transformers`` ViT module names at a fraction of the size."""

    def __init__(
        self,
        hidden: int = HIDDEN,
        intermediate: int = INTERMEDIATE,
        n_layers: int = N_LAYERS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.vit = nn.Module()
        self.vit.encoder = nn.Module()
        self.vit.encoder.layer = nn.ModuleList(
            _EncoderLayer(hidden, intermediate) for _ in range(n_layers)
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        for layer in self.vit.encoder.layer:
            x = layer(x)
        return SimpleNamespace(logits=self.classifier(x))


def _make_experiment() -> SimpleNamespace:
    model = HuggingFaceImageClassifierAdapter(_TinyViT())
    return SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            adapter_filter=BACKBONE.adapter_filter,
            head_module=BACKBONE.head_module,
            seed=0,
        ),
        num_tasks=N_TASKS,
        plugins=[],
        tb_log=SimpleNamespace(writer=None),
    )


def _members(model: nn.Module) -> list[nn.Module]:
    """The model(s) an arm actually trains: ensemble members, or the model itself."""
    if isinstance(model, EnsembleModule):
        return list(model.members)
    return [model]


@pytest.fixture(params=sorted(ARMS))
def arm_name(request) -> str:
    return request.param


class TestConstructor:
    def test_is_a_registered_dataclass_arm(self, arm_name):
        arm_cls = get_arm(arm_name)
        assert dataclasses.is_dataclass(arm_cls)
        assert issubclass(arm_cls, ArmBase)
        assert arm_cls.name == arm_name

    @pytest.mark.parametrize("lr", [1e-4, 1e-3, 5e-2])
    def test_constructs_with_overridden_lr(self, arm_name, lr):
        arm = get_arm(arm_name)(lr=lr)
        assert isinstance(arm, ArmBase)
        assert arm.lr == lr
        assert arm.name == arm_name


class TestParameterCounts:
    def test_head_is_trainable_and_backbone_is_not_fully_trainable(self, arm_name):
        arm = get_arm(arm_name)()
        experiment = _make_experiment()
        arm._build_peft(experiment)

        for member in _members(experiment.model):
            all_params = dict(member.named_parameters())
            head_params = dict(
                member.get_submodule(experiment.config.head_module).named_parameters()
            )
            assert head_params, "head module has no parameters"
            assert all(p.requires_grad for p in head_params.values())

            trainable = {n for n, p in all_params.items() if p.requires_grad}
            assert 0 < len(trainable) < len(all_params), (
                "expected a strict subset of parameters to be trainable "
                "(adapters/head only, not the full backbone)"
            )

            head_names = {
                f"{experiment.config.head_module}.{n}" for n in head_params
            }
            non_head_trainable = trainable - head_names
            if arm_name in LAZY_ADAPTER_ARMS:
                assert not non_head_trainable, (
                    f"{arm_name} adapters are populated lazily by a plugin; "
                    "none should be trainable right after construction"
                )
            else:
                assert non_head_trainable, (
                    f"{arm_name} should have trainable adapter parameters "
                    "beyond the head immediately after _build_peft"
                )


class TestForwardPassShape:
    def test_forward_pass_produces_expected_logit_shape(self, arm_name):
        arm = get_arm(arm_name)()
        experiment = _make_experiment()
        arm._build_peft(experiment)

        x = torch.randn(BATCH, HIDDEN)
        out = experiment.model(x)

        assert out.shape == (BATCH, NUM_CLASSES)
        assert torch.isfinite(out).all()
