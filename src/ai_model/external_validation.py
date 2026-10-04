"""
External validation of the trained CrossScaleDiseaseModel (runs/best.pt) on two
leaf datasets the model never saw during training:

  1. SoyNet "Camera Clicks"      — Healthy_pic / Disease_Pic
  2. India Soybean (Kotwal & Kashyap) — 1.Healthy, 2.Vein Necrosis, 3.Dry_leaf,
                                         4.Septoria_Brown_Spot, 6.Bacterial leaf Blight

Both are evaluated as binary Healthy vs. Diseased: our 5 disease classes collapse
to "Diseased". No UAV images exist for these datasets, so the UAV branch receives
a neutral input (an all-zero tensor after normalization).

The checkpoint is only read, never written.

Usage:
    python -m src.ai_model.external_validation
    python -m src.ai_model.external_validation --soynet-dir "..." --india-dir "..."
"""
import argparse
import csv
import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

from src.ai_model.config import LEAF_CLASSES, IMAGE_SIZE
from src.ai_model.data.datasets import default_transform
from src.ai_model.models.full_model import CrossScaleDiseaseModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

HEALTHY, DISEASED = 0, 1
BINARY_NAMES = {HEALTHY: "Healthy", DISEASED: "Diseased"}

SOYNET_FOLDERS = {
    "Healthy_pic": HEALTHY,
    "Disease_Pic": DISEASED,
}

# None = evaluated separately, not part of the main result
INDIA_FOLDERS = {
    "1.Healthy": HEALTHY,
    "2.Vein Necrosis": DISEASED,
    "3.Dry_leaf": None,
    "4.Septoria_Brown_Spot": DISEASED,
    "6.Bacterial leaf Blight": DISEASED,
}

# Exact 6-class check: India folder -> class name the model should predict
INDIA_EXACT_TARGETS = {
    "1.Healthy": "Healthy",
    "4.Septoria_Brown_Spot": "Septoria_brown_spot",
}

CSV_FIELDS = [
    "section", "dataset", "subset", "folder", "n",
    "n_true_healthy", "n_true_diseased",
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "tn", "fp", "fn", "tp",
    "pred_healthy_pct", "pred_diseased_pct",
    "exact_target", "exact_correct", "exact_pct",
] + [f"pred_{c}" for c in LEAF_CLASSES]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def split_tag(class_dir: Path, img: Path):
    """Single / Multi from the first subfolder under the class folder, else unsplit."""
    rel = img.relative_to(class_dir).parts
    if len(rel) > 1:
        top = rel[0].lower()
        if top.startswith("single"):
            return "Single"
        if top.startswith("multi"):
            return "Multi"
    return "unsplit"


def collect_samples(root: Path, dataset: str, folder_map: dict, tag_splits: bool):
    samples = []
    for folder, label in folder_map.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"expected class folder not found: {class_dir}")
        for p in list_images(class_dir):
            samples.append({
                "path": str(p),
                "dataset": dataset,
                "folder": folder,
                "subset": split_tag(class_dir, p) if tag_splits else "all",
                "label": label,
            })
    return samples


class ExternalLeafDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        # Same eval transform as training: Resize((224, 224)) + ToTensor + ImageNet Normalize
        self.transform = default_transform(train=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path = self.samples[i]["path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"could not read image {path}") from e
        return self.transform(img), i


@torch.no_grad()
def predict(model, samples, device, batch_size, num_workers):
    loader = DataLoader(ExternalLeafDataset(samples), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    done = 0
    for leaf, idx in loader:
        leaf = leaf.to(device)
        # No UAV images available: neutral UAV input = zero tensor after normalization
        uav = torch.zeros(leaf.size(0), 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        logits, _, _ = model(leaf, uav)
        probs = logits.softmax(dim=-1)
        conf, pred = probs.max(dim=-1)
        for j, k in enumerate(idx.tolist()):
            p = pred[j].item()
            samples[k]["pred_class"] = LEAF_CLASSES[p]
            samples[k]["pred_binary"] = HEALTHY if p == LEAF_CLASSES.index("Healthy") else DISEASED
            samples[k]["confidence"] = conf[j].item()
        done += leaf.size(0)
        print(f"  {done}/{len(samples)}", end="\r", flush=True)
    print()


def class_counts(rows):
    return {f"pred_{c}": sum(r["pred_class"] == c for r in rows) for c in LEAF_CLASSES}


def binary_metrics(rows):
    y_true = np.array([r["label"] for r in rows])
    y_pred = np.array([r["pred_binary"] for r in rows])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[DISEASED], average=None, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[HEALTHY, DISEASED]).ravel()
    return {
        "n": len(rows),
        "n_true_healthy": int((y_true == HEALTHY).sum()),
        "n_true_diseased": int((y_true == DISEASED).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": prec[0],
        "recall": rec[0],
        "f1": f1[0],
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pred_split(rows):
    n = len(rows)
    h = sum(r["pred_binary"] == HEALTHY for r in rows)
    return {"n": n, "pred_healthy_pct": 100 * h / n, "pred_diseased_pct": 100 * (n - h) / n}


def subsets(rows, names=("Single", "Multi")):
    """Yield (subset_name, rows) for each named split plus 'Combined' (all rows)."""
    for name in names:
        sub = [r for r in rows if r["subset"] == name]
        if sub:
            yield name, sub
    yield "Combined", rows


def run_analyses(soynet, india):
    out = []

    # Per-folder image counts and 6-class prediction breakdown
    for ds_rows in (soynet, india):
        keys = sorted({(r["dataset"], r["folder"], r["subset"]) for r in ds_rows})
        for ds, folder, subset in keys:
            rows = [r for r in ds_rows if r["folder"] == folder and r["subset"] == subset]
            out.append({"section": "folder_breakdown", "dataset": ds, "subset": subset,
                        "folder": folder, **pred_split(rows), **class_counts(rows)})

    # Main binary result: SoyNet
    out.append({"section": "binary_main", "dataset": "SoyNet", "subset": "all",
                "folder": "Healthy_pic + Disease_Pic", **binary_metrics(soynet)})

    # Main binary result: India (Dry_leaf excluded)
    india_main = [r for r in india if r["label"] is not None]
    for name, rows in subsets(india_main):
        folders = sorted({r["folder"] for r in rows})
        out.append({"section": "binary_main", "dataset": "India Soybean", "subset": name,
                    "folder": " + ".join(folders), **binary_metrics(rows)})

    # Dry_leaf, reported separately
    dry = [r for r in india if r["folder"] == "3.Dry_leaf"]
    for name, rows in subsets(dry):
        out.append({"section": "dry_leaf", "dataset": "India Soybean", "subset": name,
                    "folder": "3.Dry_leaf", **pred_split(rows), **class_counts(rows)})

    # Exact 6-class check
    for folder, target in INDIA_EXACT_TARGETS.items():
        fr = [r for r in india if r["folder"] == folder]
        for name, rows in subsets(fr):
            hit = sum(r["pred_class"] == target for r in rows)
            out.append({"section": "exact_class", "dataset": "India Soybean", "subset": name,
                        "folder": folder, "n": len(rows), "exact_target": target,
                        "exact_correct": hit, "exact_pct": 100 * hit / len(rows),
                        **class_counts(rows)})
    return out


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def _f(x):
    return f"{x:.3f}"


def _pct(x):
    return f"{x:.1f}%"


def write_markdown(rows, meta, path):
    sec = lambda s: [r for r in rows if r["section"] == s]
    md = []
    md.append("# External validation — CrossScaleDiseaseModel\n")
    md.append(f"Generated {meta['date']} · checkpoint `{meta['checkpoint']}` · config `{meta['config']}` · "
              f"device `{meta['device']}`\n")
    md.append("Binary task: **Healthy vs. Diseased**. The model's 5 disease classes "
              f"({', '.join(LEAF_CLASSES[1:])}) are collapsed to *Diseased*; *Healthy* stays *Healthy*. "
              "Precision, recall and F1 treat **Diseased as the positive class**.\n")
    md.append("> **UAV input:** these datasets have no UAV images. Every leaf image was paired with a "
              "neutral UAV input (an all-zero tensor after normalization). See *Limitations*.\n")

    md.append("## Image counts per folder\n")
    body = []
    for r in sec("folder_breakdown"):
        body.append([r["dataset"], r["folder"], r["subset"], r["n"]])
    md.append(_table(["Dataset", "Folder", "Split", "Images"], body) + "\n")
    md.append("SoyNet's mobile-phone images were **excluded** because they have no labels; only the "
              "labelled *Camera Clicks* set is used.\n")

    md.append("## Main result — Healthy vs. Diseased\n")
    body = []
    for r in sec("binary_main"):
        body.append([r["dataset"], r["subset"], r["n"], r["n_true_healthy"], r["n_true_diseased"],
                     _f(r["accuracy"]), _f(r["balanced_accuracy"]), _f(r["precision"]),
                     _f(r["recall"]), _f(r["f1"])])
    md.append(_table(["Dataset", "Split", "N", "True Healthy", "True Diseased", "Accuracy",
                      "Balanced acc.", "Precision", "Recall", "F1"], body) + "\n")
    md.append("India Soybean excludes *3.Dry_leaf* (reported separately). *6.Bacterial leaf Blight* "
              "has no Single/Multi split, so it only appears in the **Combined** row.\n")

    md.append("### Confusion matrices (rows = true, columns = predicted)\n")
    for r in sec("binary_main"):
        md.append(f"**{r['dataset']} — {r['subset']}** (n = {r['n']})\n")
        md.append(_table(["", "Pred Healthy", "Pred Diseased"],
                         [["True Healthy", r["tn"], r["fp"]],
                          ["True Diseased", r["fn"], r["tp"]]]) + "\n")

    md.append("## India Soybean — Dry_leaf (not in the main result)\n")
    body = [[r["subset"], r["n"], _pct(r["pred_healthy_pct"]), _pct(r["pred_diseased_pct"])]
            for r in sec("dry_leaf")]
    md.append(_table(["Split", "N", "Predicted Healthy", "Predicted Diseased"], body) + "\n")
    md.append("Dry leaf may be senescence or drought stress rather than a disease, so there is no "
              "correct binary label for it. These numbers show how the model behaves on it, not "
              "whether it is right.\n")

    md.append("## Exact-class check (6-class)\n")
    body = [[r["folder"], r["subset"], r["n"], r["exact_target"], r["exact_correct"], _pct(r["exact_pct"])]
            for r in sec("exact_class")]
    md.append(_table(["Folder", "Split", "N", "Expected class", "Exact hits", "Exact %"], body) + "\n")

    md.append("## Predicted 6-class distribution per folder\n")
    body = [[r["dataset"], r["folder"], r["subset"], r["n"]] + [r[f"pred_{c}"] for c in LEAF_CLASSES]
            for r in sec("folder_breakdown")]
    md.append(_table(["Dataset", "Folder", "Split", "N"] + LEAF_CLASSES, body) + "\n")

    md.append("## Limitations\n")
    md.append("\n".join([
        "- **No UAV input.** The model is a leaf + UAV fusion network trained with real UAV images. "
        "Here the UAV branch received an all-zero tensor after normalization (equivalent to a flat "
        "image of the ImageNet mean colour), which the model never saw in training. These results "
        "measure the leaf pathway under an out-of-distribution UAV input, **not** the full system as "
        "designed, and may understate or misrepresent its performance.",
        "- **SoyNet mobile images excluded.** They have no labels, so only the labelled Camera Clicks "
        "photos were used.",
        "- **Binary collapse.** The external label sets don't match our 6 classes, so most of the "
        "evaluation is Healthy vs. Diseased only. A \"Diseased\" hit can come from the wrong disease "
        "class; the exact-class check covers only Healthy and Septoria brown spot.",
        "- **Diseases outside our label set.** Vein necrosis and bacterial leaf blight are not among "
        "our trained classes; counting them as correct whenever the model predicts any disease is a "
        "generous assumption.",
        "- **Dry_leaf has no ground truth** for this task and is kept out of the main metrics.",
        "- **6.Bacterial leaf Blight has no Single/Multi split**, so Single and Multi results for "
        "India exclude it and are not directly comparable to Combined.",
        "- **Class imbalance.** SoyNet is about 6:1 Diseased to Healthy and India about 2:1, so plain "
        "accuracy is inflated by always predicting Diseased; read balanced accuracy alongside it.",
        "- **Preprocessing.** Full-size photos are resized straight to 224×224 like in training "
        "(no crop, aspect ratio not preserved, EXIF orientation not applied).",
        "- **Domain shift.** Different cameras, backgrounds, lighting and regions from the training data; "
        "single runs with no confidence intervals.",
    ]) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main(args):
    cfg = load_config(args.config)
    device = get_device(args.device)

    soynet = collect_samples(Path(args.soynet_dir), "SoyNet", SOYNET_FOLDERS, tag_splits=False)
    india = collect_samples(Path(args.india_dir), "India Soybean", INDIA_FOLDERS, tag_splits=True)
    print(f"SoyNet: {len(soynet)} images, India Soybean: {len(india)} images, device: {device}")

    model = CrossScaleDiseaseModel(
        num_classes=cfg["model"]["num_classes"],
        leaf_backbone=cfg["model"]["leaf_backbone"],
        uav_backbone=cfg["model"]["uav_backbone"],
        fusion_dim=cfg["model"]["fusion_dim"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    for name, samples in (("SoyNet", soynet), ("India Soybean", india)):
        print(f"Predicting {name}...")
        predict(model, samples, device, args.batch_size, args.num_workers)

    rows = run_analyses(soynet, india)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "external_validation.csv")
    write_markdown(rows, {
        "date": datetime.date.today().isoformat(),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": device,
    }, out_dir / "external_validation.md")

    for r in rows:
        if r["section"] == "binary_main":
            print(f"{r['dataset']:14s} {r['subset']:9s} n={r['n']:5d} acc={r['accuracy']:.3f} "
                  f"bal_acc={r['balanced_accuracy']:.3f} f1={r['f1']:.3f}")
    print(f"Wrote {out_dir / 'external_validation.csv'} and {out_dir / 'external_validation.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External validation on SoyNet and India Soybean leaf datasets")
    parser.add_argument("--soynet-dir", default="dataset/external/Camera Clicks")
    parser.add_argument("--india-dir", default="dataset/external/Soyabean leaf")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/best.pt")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cuda or cpu")
    main(parser.parse_args())
