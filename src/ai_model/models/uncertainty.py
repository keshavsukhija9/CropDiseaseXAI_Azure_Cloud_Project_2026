import torch
import torch.nn.functional as F


def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, leaf, uav, passes: int = 20):
    """Module 4: MC-Dropout uncertainty -> auto-accept vs agronomist review."""
    enable_mc_dropout(model)
    probs_list = []
    for _ in range(passes):
        logits, *_ = model(leaf, uav)
        probs_list.append(F.softmax(logits, dim=-1))
    probs = torch.stack(probs_list, dim=0)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-8))).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(mean_probs.shape[-1])))
    return mean_probs, entropy / max_entropy


def decide(normalized_entropy: float, low_thr: float = 0.35, high_thr: float = 0.65) -> str:
    if normalized_entropy <= low_thr:
        return "Automatically accepted"
    if normalized_entropy >= high_thr:
        return "Agronomist review required"
    return "Flagged for spot-check"
