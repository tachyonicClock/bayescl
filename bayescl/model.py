from torch import Tensor, nn
from typing import TYPE_CHECKING
from transformers import AutoModelForImageClassification
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
if TYPE_CHECKING:
    from bayescl.experiment import ExperimentConfig

class ResNetHuggingFaceAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits


def get_model(config: "ExperimentConfig", num_classes: int) -> nn.Module:
    model: nn.Module = AutoModelForImageClassification.from_pretrained(
        config.backbone_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Allows for different number of classes
    )
    if config.freeze_backbone:
        model.requires_grad_(False)  # Freeze the backbone
        model.classifier.requires_grad_(True)  # Unfreeze the classifier layer

    if isinstance(model, ResNetForImageClassification):
        return ResNetHuggingFaceAdapter(model)
    raise NotImplementedError(f"Got unsupported model type: {type(model)}")
