import numpy as np


def binarize(saliency, percentile=80):
    thr = np.percentile(saliency, percentile)
    return (saliency >= thr).astype(np.uint8)


def iou(saliency_mask, disease_mask):
    inter = np.logical_and(saliency_mask, disease_mask).sum()
    union = np.logical_or(saliency_mask, disease_mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def pointing_game_hit(saliency, disease_mask):
    peak_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    return bool(disease_mask[peak_idx])


def evaluate_localization(saliency_maps: dict, disease_mask, percentile=80):
    results = {}
    for name, sal in saliency_maps.items():
        mask = binarize(sal, percentile)
        results[name] = {"iou": iou(mask, disease_mask), "pointing_game_hit": pointing_game_hit(sal, disease_mask)}
    return results
