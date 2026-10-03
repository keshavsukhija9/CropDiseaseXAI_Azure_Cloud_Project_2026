import timm
import torch.nn as nn


class UAVEncoder(nn.Module):
    """Macro/field-level: ViT over the UAV image."""

    def __init__(self, backbone: str = "vit_small_patch16_224", embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))
