import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize
from PIL import Image

from src.ai_model.robustness.perturbations import PERTURBATIONS
from src.ai_model.config import IMAGE_SIZE


def _to_model_input(np_img):
    img = Image.fromarray(np_img).resize((IMAGE_SIZE, IMAGE_SIZE))
    t = to_tensor(img)
    t = normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t.unsqueeze(0)


@torch.no_grad()
def evaluate_robustness(model, leaf_np_images, labels, uav_fixed, device):
    results = {}
    conditions = {"clean": lambda x: x, **PERTURBATIONS}
    for name, fn in conditions.items():
        correct = 0
        for img, label in zip(leaf_np_images, labels):
            perturbed = fn(img)
            x = _to_model_input(perturbed).to(device)
            logits, _, _ = model(x, uav_fixed)
            pred = logits.argmax(dim=-1).item()
            correct += int(pred == label)
        results[name] = correct / max(len(labels), 1)
    return results
