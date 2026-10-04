"""
Runs the real trained checkpoint against the available dataset and saves
confusion matrix, classification report, and robustness charts to results/.

Uses the full dataset if present under dataset/mh_soyahealthvision_leaf and
dataset/mh_soyahealthvision_uav, otherwise falls back to the 30-image sample
set under dataset/samples/.
"""
import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

from src.ai_model.config import LEAF_CLASSES
from src.ai_model.data.datasets import PairedFusionDataset
from src.ai_model.models.full_model import CrossScaleDiseaseModel
from src.ai_model.robustness.evaluate_robustness import evaluate_robustness

os.makedirs("results", exist_ok=True)

FULL_LEAF = "dataset/mh_soyahealthvision_leaf"
FULL_UAV = "dataset/mh_soyahealthvision_uav"
SAMPLE_LEAF = "dataset/samples/leaf"
SAMPLE_UAV = "dataset/samples/uav"

if os.path.isdir(FULL_LEAF) and os.path.isdir(FULL_UAV):
    leaf_dir, uav_dir = FULL_LEAF, FULL_UAV
    print("Using full dataset")
else:
    leaf_dir, uav_dir = SAMPLE_LEAF, SAMPLE_UAV
    print("Full dataset not found, using 30-image sample set instead")

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

model = CrossScaleDiseaseModel(
    num_classes=cfg["model"]["num_classes"],
    leaf_backbone=cfg["model"]["leaf_backbone"],
    uav_backbone=cfg["model"]["uav_backbone"],
    fusion_dim=cfg["model"]["fusion_dim"],
    pretrained=False,
).to(device)
model.load_state_dict(torch.load("runs/best.pt", map_location=device))
model.eval()

ds = PairedFusionDataset(leaf_dir, uav_dir, train=False)
loader = DataLoader(ds, batch_size=4)

y_true, y_pred = [], []
with torch.no_grad():
    for batch in loader:
        leaf = batch["leaf"].to(device)
        uav = batch["uav"].to(device)
        logits, _, _ = model(leaf, uav)
        y_pred += logits.argmax(dim=-1).cpu().tolist()
        y_true += batch["label"].tolist()

present_labels = sorted(set(y_true) | set(y_pred))
present_names = [LEAF_CLASSES[i] for i in present_labels]

print(classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, digits=3))

cm = confusion_matrix(y_true, y_pred, labels=present_labels)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(present_names))); ax.set_xticklabels(present_names, rotation=45, ha="right")
ax.set_yticks(range(len(present_names))); ax.set_yticklabels(present_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion matrix (n=%d, live run on trained checkpoint)" % len(y_true))
thresh = cm.max() / 2 if cm.max() > 0 else 1
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=10)
fig.colorbar(im, ax=ax, label="count")
fig.tight_layout()
fig.savefig("results/confusion_matrix_live.png")
plt.close(fig)
print("saved results/confusion_matrix_live.png")

print("Running robustness suite on leaf samples...")
rng = np.random.default_rng(42)
sample_count = min(30, len(ds))
indices = rng.choice(len(ds), size=sample_count, replace=False)
leaf_images, labels = [], []
for i in indices:
    sample = ds[int(i)]
    img = Image.open(sample["leaf_path"]).convert("RGB").resize((224, 224))
    leaf_images.append(np.array(img))
    labels.append(sample["label"])
uav_fixed = ds[int(indices[0])]["uav"].unsqueeze(0).to(device)

robustness_results = evaluate_robustness(model, leaf_images, labels, uav_fixed, device)

conditions = list(robustness_results.keys())
accuracies = list(robustness_results.values())
order = np.argsort(accuracies)[::-1]
conditions_sorted = [conditions[i] for i in order]
accuracies_sorted = [accuracies[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2ca02c" if c == "clean" else
          "#d62728" if a < 0.5 else "#ff7f0e"
          for c, a in zip(conditions_sorted, accuracies_sorted)]
bars = ax.bar(conditions_sorted, accuracies_sorted, color=colors)
ax.set_ylabel("Accuracy")
ax.set_title("Robustness under perturbations, live run (n=%d)" % sample_count)
ax.set_xticklabels(conditions_sorted, rotation=30, ha="right")
for bar, val in zip(bars, accuracies_sorted):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, "%.3f" % val, ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results/robustness_chart_live.png")
plt.close(fig)
print("saved results/robustness_chart_live.png")

print("Done. Open the results folder to view the images.")
