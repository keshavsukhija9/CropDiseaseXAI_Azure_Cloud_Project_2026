"""
Dataset loaders for MH-SoyaHealthVision — two parts per Dr. Priya V's document:
leaf (micro/ground-level) and UAV (macro/field-level).
"""
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.ai_model.config import LEAF_CLASSES, UAV_CLASSES, IMAGE_SIZE


def default_transform(train: bool = False):
    ops = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
    ops += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(ops)


class _ImageFolderByClass(Dataset):
    def __init__(self, root: str, classes: list, train: bool = False):
        self.root = Path(root)
        self.classes = classes
        self.samples = []
        for idx, cls in enumerate(classes):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.jpg"):
                self.samples.append((p, idx))
        self.transform = default_transform(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, str(path)


class LeafDataset(_ImageFolderByClass):
    def __init__(self, root: str, train: bool = False):
        super().__init__(root, LEAF_CLASSES, train)


class UAVDataset(_ImageFolderByClass):
    def __init__(self, root: str, train: bool = False):
        super().__init__(root, UAV_CLASSES, train)


class PairedFusionDataset(Dataset):
    """
    Pairs each leaf sample with a UAV sample. Where the UAV folder set
    doesn't cover a leaf class (Frogeye, Septoria — UAV dataset only has
    4 folders per the document), falls back to a random UAV sample so
    every leaf image still gets *some* field-level context at training time.
    """

    def __init__(self, leaf_root: str, uav_root: str, train: bool = False):
        self.leaf = LeafDataset(leaf_root, train)
        self.uav = UAVDataset(uav_root, train)
        self.uav_by_leaf_class_name = {}
        for i, (_, label) in enumerate(self.uav.samples):
            cls_name = UAV_CLASSES[label]
            self.uav_by_leaf_class_name.setdefault(cls_name, []).append(i)

    def __len__(self):
        return len(self.leaf)

    def __getitem__(self, i):
        leaf_img, label, leaf_path = self.leaf[i]
        cls_name = LEAF_CLASSES[label]
        candidates = self.uav_by_leaf_class_name.get(cls_name)
        if candidates:
            uav_idx = random.choice(candidates)
        else:
            uav_idx = random.randrange(len(self.uav))
        uav_img, _, uav_path = self.uav[uav_idx]
        return {
            "leaf": leaf_img,
            "uav": uav_img,
            "label": label,
            "leaf_path": leaf_path,
            "uav_path": uav_path,
        }
