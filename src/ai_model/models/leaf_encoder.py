import timm
import torch.nn as nn


class LeafEncoder(nn.Module):
    """Micro-level: CNN over the leaf image."""

    def __init__(self, backbone: str = "mobilenetv2_100", embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))
