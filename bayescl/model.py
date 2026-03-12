from avalanche.models import SimpleMLP
from torch import Tensor, nn
from transformers import AutoModelForImageClassification
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

from bayescl.config import BasicModelConfig, Config, HuggingFaceModelConfig


class ResNetHuggingFaceAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits


def _get_basic_model(model_cfg: BasicModelConfig, num_classes: int) -> nn.Module:
    if model_cfg.name == "SimpleMLP":
        return SimpleMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported basic model: {model_cfg.name}")


def _get_huggingface_model(
    model_cfg: HuggingFaceModelConfig, num_classes: int
) -> nn.Module:
    model: nn.Module = AutoModelForImageClassification.from_pretrained(
        model_cfg.name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Allows for different number of classes
    )
    if model_cfg.freeze_backbone:
        model.requires_grad_(False)  # Freeze the backbone
        model.classifier.requires_grad_(True)  # Unfreeze the classifier layer

    if isinstance(model, ResNetForImageClassification):
        return ResNetHuggingFaceAdapter(model)
    else:
        raise NotImplementedError(f"Got unsupported model type: {type(model)}")


def get_model(cfg: Config, num_classes: int) -> nn.Module:
    if cfg.model.type == "basic":
        return _get_basic_model(cfg.model, num_classes)
    elif cfg.model.type == "huggingface":
        return _get_huggingface_model(cfg.model, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
