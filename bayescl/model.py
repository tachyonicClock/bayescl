from typing import override

import torch
from avalanche.models import SimpleMLP
from torch import Tensor, nn
from transformers import AutoModelForImageClassification
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2ForImageClassification,
)

from bayescl.config import BasicModelConfig, Config, HuggingFaceModelConfig
from bayescl.methods.l2p import L2PViT
from bayescl.models.fcg import SimpleFCGMLP


class HuggingFaceAdapter(L2PViT, nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        if isinstance(model, Dinov2ForImageClassification):
            self.embed_dim = model.dinov2.config.hidden_size  # type: ignore
            self.vit = model.dinov2  # type: ignore
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
    def get_embedding_size(self) -> int:
        return self.embed_dim

    @override
    def get_patch_embed(self, pixels: Tensor) -> Tensor:
        return self.vit.embeddings(pixels)

    @override
    def forward_encoder(self, prompts: Tensor, patch_embed: Tensor) -> Tensor:
        cls_token = patch_embed[:, :1, :]
        other_patches = patch_embed[:, 1:, :]

        # Add CLS position embedding to prompts
        cls_pos_embedding = self.vit.embeddings.position_embeddings[:, :1]
        prompts = prompts + cls_pos_embedding

        embedding = torch.cat([cls_token, prompts, other_patches], dim=1)
        sequence_output = self.vit.encoder(embedding).last_hidden_state
        sequence_output = self.vit.layernorm(sequence_output)
        return sequence_output

    @override
    @torch.no_grad()
    def forward_query(self, patch_embedding: Tensor) -> Tensor:
        # Return [CLS] token embedding
        sequence_output = self.vit.encoder(patch_embedding).last_hidden_state
        sequence_output = self.vit.layernorm(sequence_output)
        return sequence_output[:, 0, :]


def _get_basic_model(model_cfg: BasicModelConfig, num_classes: int) -> nn.Module:
    if model_cfg.name == "SimpleMLP":
        return SimpleMLP(num_classes=num_classes)
    elif model_cfg.name == "SimpleFCGMLP":
        return SimpleFCGMLP(
            in_features=28 * 28, out_features=num_classes, hidden_features=16
        )
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
