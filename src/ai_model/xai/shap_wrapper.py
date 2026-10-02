import numpy as np
import shap
import torch


def run_shap(model, leaf_batch, uav_fixed_single, class_idx: int, n_background: int = 8):
    background = leaf_batch[:n_background]

    class Wrapper(torch.nn.Module):
        def __init__(self, base, uav_single):
            super().__init__()
            self.base = base
            self.uav_single = uav_single

        def forward(self, leaf_x):
            uav_rep = self.uav_single.expand(leaf_x.shape[0], -1, -1, -1)
            logits, _, _ = self.base(leaf_x, uav_rep)
            return logits

    wrapped = Wrapper(model, uav_fixed_single)
    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(leaf_batch)
    return np.abs(shap_values[..., class_idx]).sum(axis=1) if shap_values.ndim == 5 else np.abs(shap_values[class_idx]).sum(axis=1)
