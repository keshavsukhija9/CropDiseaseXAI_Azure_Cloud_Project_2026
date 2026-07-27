import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import json
import os

DATA_DIR = "../../dataset/raw/plantvillage_dataset/color"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoint.pt"
CLASS_MAP_PATH = "class_to_idx.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_classes = len(full_dataset.classes)
print(f"Found {len(full_dataset)} images across {num_classes} classes")

with open(CLASS_MAP_PATH, "w") as f:
    json.dump(full_dataset.class_to_idx, f, indent=2)

val_size = int(0.15 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- Model: transfer learning on MobileNetV2 ----------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- Train ----------
def run_epoch(loader, train_mode=True):
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if train_mode:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        if train_mode and batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{len(loader)} | running_acc={correct/total:.4f}")
    return total_loss / total, correct / total

if __name__ == "__main__":
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc = run_epoch(val_loader, train_mode=False)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_acc:.4f}. Checkpoint: {CHECKPOINT_PATH}")
