from typing import override

from avalanche.models import SimpleMLP
from torch import Tensor, nn
from transformers import AutoModelForImageClassification
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2ForImageClassification,
)

from bayescl.config import BasicModelConfig, Config, HuggingFaceModelConfig
from bayescl.methods.l2p import Backbone


class HuggingFaceAdapter(Backbone, nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        if isinstance(model, Dinov2ForImageClassification):
            self.embed_dim = model.dinov2.config.hidden_size  # type: ignore
        else:
            raise NotImplementedError(f"Got unsupported model type: {type(model)}")

    def get_encoder(self) -> nn.Module:
        if hasattr(self.model, "vit"):
            return self.model.vit  # type: ignore
        elif hasattr(self.model, "dinov2"):
            return self.model.dinov2  # type: ignore
        else:
            raise ValueError("Unsupported model architecture for encoder extraction.")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits

    @override
    def embed_patches(self, pixel_values: Tensor) -> Tensor:
        if isinstance(self.model, Dinov2ForImageClassification):
            return self.model.dinov2.embeddings(pixel_values=pixel_values)
        else:
            raise NotImplementedError("embed_patches not implemented for this model.")

    @override
    def forward_encoder(self, patch_embeddings: Tensor) -> Tensor:
        if isinstance(self.model, Dinov2ForImageClassification):
            return self.model.dinov2.encoder(patch_embeddings).last_hidden_state  # type: ignore
        else:
            raise NotImplementedError("forward_encoder not implemented for this model.")

    @override
    def forward_query(self, patch_embeddings: Tensor) -> Tensor:
        return self.forward_encoder(patch_embeddings)[:, 0]


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

    return HuggingFaceAdapter(model)


def get_model(cfg: Config, num_classes: int) -> nn.Module:
    if cfg.model.type == "basic":
        return _get_basic_model(cfg.model, num_classes)
    elif cfg.model.type == "huggingface":
        return _get_huggingface_model(cfg.model, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
