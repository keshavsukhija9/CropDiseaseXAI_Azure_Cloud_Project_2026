import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _leaf_target_layer(model):
    backbone = model.leaf_encoder.backbone
    if hasattr(backbone, "conv_head"):
        return [backbone.conv_head]
    return [list(backbone.children())[-2]]


def run_gradcam(model, leaf, uav, class_idx: int, plus_plus: bool = False):
    target_layers = _leaf_target_layer(model)
    cam_cls = GradCAMPlusPlus if plus_plus else GradCAM

    class Wrapper(torch.nn.Module):
        def __init__(self, base, uav_fixed):
            super().__init__()
            self.base = base
            self.uav_fixed = uav_fixed

        def forward(self, leaf_x):
            logits, _, _ = self.base(leaf_x, self.uav_fixed)
            return logits

    wrapped = Wrapper(model, uav)
    cam = cam_cls(model=wrapped, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    return cam(input_tensor=leaf, targets=targets)
