import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from src.ai_model.data.datasets import PairedFusionDataset
from src.ai_model.models.full_model import CrossScaleDiseaseModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(cfg_path):
    cfg = load_config(cfg_path)
    torch.manual_seed(cfg["train"]["seed"])
    device = get_device()
    print("device:", device)

    full_ds = PairedFusionDataset(cfg["data"]["leaf_dir"], cfg["data"]["uav_dir"], train=True)
    val_size = int(0.15 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], num_workers=0)

    model = CrossScaleDiseaseModel(
        num_classes=cfg["model"]["num_classes"],
        leaf_backbone=cfg["model"]["leaf_backbone"],
        uav_backbone=cfg["model"]["uav_backbone"],
        fusion_dim=cfg["model"]["fusion_dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            leaf = batch["leaf"].to(device)
            uav = batch["uav"].to(device)
            label = batch["label"].to(device)
            logits, severity_frac, _ = model(leaf, uav)
            loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                leaf = batch["leaf"].to(device)
                uav = batch["uav"].to(device)
                label = batch["label"].to(device)
                logits, _, _ = model(leaf, uav)
                pred = logits.argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{cfg['train']['epochs']}  loss={running_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "runs/best.pt")
            print(f"  -> saved new best (val_acc={val_acc:.4f})")

    print("training done. best val_acc:", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
