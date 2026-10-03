import torch.nn as nn


class SeverityHead(nn.Module):
    """Module 2: image-derived severity index (NOT expert ground truth, per the document)."""

    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, fused_embed):
        return self.net(fused_embed).squeeze(-1)

    @staticmethod
    def bucket(pct: float, low_max: float = 10.0, moderate_max: float = 35.0) -> str:
        if pct <= low_max:
            return "Low"
        if pct <= moderate_max:
            return "Moderate"
        return "High"
