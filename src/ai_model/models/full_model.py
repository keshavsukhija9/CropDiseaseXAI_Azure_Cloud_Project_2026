import torch.nn as nn
from src.ai_model.models.leaf_encoder import LeafEncoder
from src.ai_model.models.uav_encoder import UAVEncoder
from src.ai_model.models.fusion import CrossScaleAttentionFusion
from src.ai_model.models.severity import SeverityHead


class CrossScaleDiseaseModel(nn.Module):
    def __init__(self, num_classes: int, leaf_backbone: str, uav_backbone: str,
                 fusion_dim: int = 256, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.leaf_encoder = LeafEncoder(leaf_backbone, fusion_dim, pretrained)
        self.uav_encoder = UAVEncoder(uav_backbone, fusion_dim, pretrained)
        self.fusion = CrossScaleAttentionFusion(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.severity_head = SeverityHead(fusion_dim)

    def forward(self, leaf, uav):
        leaf_embed = self.leaf_encoder(leaf)
        uav_embed = self.uav_encoder(uav)
        fused = self.dropout(self.fusion(leaf_embed, uav_embed))
        return self.classifier(fused), self.severity_head(fused), fused
