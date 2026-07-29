import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np

DATA_DIR = "../../dataset/raw/plantvillage_dataset/color"
IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoint.pt"
CLASS_MAP_PATH = "class_to_idx.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open(CLASS_MAP_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Same seed as training so we evaluate on the SAME held-out val split
torch.manual_seed(42)
val_size = int(0.15 * len(full_dataset))
train_size = len(full_dataset) - val_size
_, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print(f"\nEvaluated on {len(all_labels)} held-out validation images\n")
print(classification_report(all_labels, all_preds, target_names=[idx_to_class[i] for i in range(num_classes)], digits=4, zero_division=0))

# Flag any class with suspiciously perfect (100%) precision+recall - possible leakage signal
report = classification_report(all_labels, all_preds, target_names=[idx_to_class[i] for i in range(num_classes)], output_dict=True, zero_division=0)
perfect_classes = [cls for cls, metrics in report.items() if isinstance(metrics, dict) and metrics.get("recall") == 1.0 and metrics.get("precision") == 1.0]
print(f"\nClasses with 100% precision AND recall ({len(perfect_classes)}/{num_classes}):")
for c in perfect_classes:
    print(f"  - {c}")
print("\nNote: a high count of perfect classes on a dataset this size can indicate near-duplicate")
print("images leaking between train/val splits, not necessarily true generalisation.")
