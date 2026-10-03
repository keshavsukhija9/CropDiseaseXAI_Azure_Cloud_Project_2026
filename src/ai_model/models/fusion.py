import torch
import torch.nn as nn


class CrossScaleAttentionFusion(nn.Module):
    """Module 1: leaf and UAV embeddings attend over each other, then concat."""

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.leaf_to_uav = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.uav_to_leaf = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim * 2)
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, leaf_embed, uav_embed):
        leaf_q = leaf_embed.unsqueeze(1)
        uav_q = uav_embed.unsqueeze(1)
        leaf_attended, _ = self.leaf_to_uav(leaf_q, uav_q, uav_q)
        uav_attended, _ = self.uav_to_leaf(uav_q, leaf_q, leaf_q)
        fused = torch.cat([leaf_attended.squeeze(1), uav_attended.squeeze(1)], dim=-1)
        return self.out(self.norm(fused))
