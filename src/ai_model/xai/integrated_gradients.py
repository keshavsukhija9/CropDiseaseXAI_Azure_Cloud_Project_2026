import torch
from captum.attr import IntegratedGradients


def run_integrated_gradients(model, leaf, uav, class_idx: int, steps: int = 32):
    def forward_leaf_only(leaf_x):
        uav_expanded = uav.expand(leaf_x.shape[0], -1, -1, -1)
        logits, _, _ = model(leaf_x, uav_expanded)
        return logits

    ig = IntegratedGradients(forward_leaf_only)
    baseline = torch.zeros_like(leaf)
    attributions = ig.attribute(leaf, baselines=baseline, target=class_idx, n_steps=steps)
    saliency = attributions.abs().sum(dim=1)
    saliency = (saliency - saliency.amin(dim=(1, 2), keepdim=True)) / (
        saliency.amax(dim=(1, 2), keepdim=True) - saliency.amin(dim=(1, 2), keepdim=True) + 1e-8
    )
    return saliency
