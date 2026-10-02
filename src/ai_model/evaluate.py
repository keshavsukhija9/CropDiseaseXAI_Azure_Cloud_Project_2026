import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.ai_model.data.datasets import PairedFusionDataset
from src.ai_model.models.full_model import CrossScaleDiseaseModel
from src.ai_model.config import LEAF_CLASSES


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main(cfg_path, checkpoint):
    cfg = load_config(cfg_path)
    device = get_device()

    ds = PairedFusionDataset(cfg["data"]["leaf_dir"], cfg["data"]["uav_dir"], train=False)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"])

    model = CrossScaleDiseaseModel(
        num_classes=cfg["model"]["num_classes"],
        leaf_backbone=cfg["model"]["leaf_backbone"],
        uav_backbone=cfg["model"]["uav_backbone"],
        fusion_dim=cfg["model"]["fusion_dim"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    for batch in loader:
        leaf = batch["leaf"].to(device)
        uav = batch["uav"].to(device)
        logits, _, _ = model(leaf, uav)
        y_pred += logits.argmax(dim=-1).cpu().tolist()
        y_true += batch["label"].tolist()

    print(classification_report(y_true, y_pred, target_names=LEAF_CLASSES, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(LEAF_CLASSES)
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/best.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
