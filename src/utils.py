"""
src/utils.py — Reproducibility, device management, and data loading.

Supported datasets (none used in course notebooks 01-11):
  - KMNIST   (28×28 grayscale, 10 classes of Kuzushiji characters)
  - SVHN     (32×32 RGB, 10 classes of street-view house numbers)
  - STL-10   (96×96 RGB, 10 classes — few-shot regime)
  - EuroSAT  (64×64 RGB satellite, 10 land-use classes)
"""

import os
import random
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Normalization constants ─────────────────────────────────────────────────

KMNIST_MEAN, KMNIST_STD = (0.1918,), (0.3483,)
SVHN_MEAN, SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
STL10_MEAN, STL10_STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
EUROSAT_MEAN, EUROSAT_STD = (0.3444, 0.3803, 0.4078), (0.2035, 0.1367, 0.1156)


# ── Transforms ──────────────────────────────────────────────────────────────

def _get_transforms(dataset_name: str):
    """Return (train_transform, test_transform)."""
    name = dataset_name.lower()

    if name == "kmnist":
        tf = T.Compose([T.ToTensor(), T.Normalize(KMNIST_MEAN, KMNIST_STD)])
        return tf, tf
    elif name == "svhn":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(SVHN_MEAN, SVHN_STD)])
        return train_tf, test_tf
    elif name == "stl10":
        train_tf = T.Compose([
            T.RandomCrop(96, padding=12),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(STL10_MEAN, STL10_STD),
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(STL10_MEAN, STL10_STD)])
        return train_tf, test_tf
    elif name == "eurosat":
        train_tf = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(EUROSAT_MEAN, EUROSAT_STD),
        ])
        test_tf = T.Compose([
            T.Resize(64),
            T.ToTensor(),
            T.Normalize(EUROSAT_MEAN, EUROSAT_STD),
        ])
        return train_tf, test_tf
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Use 'kmnist', 'svhn', 'stl10', or 'eurosat'.")


# ── DataLoaders ─────────────────────────────────────────────────────────────

def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 0,
    data_root: str = "../data",  # MODIFICADO: Apunta a la raíz del proyecto para evitar duplicados en ipynb/
):
    """
    Return (train_loader, test_loader) for the specified dataset.
    Supported: 'kmnist', 'svhn', 'stl10', 'eurosat'.
    """
    name = dataset_name.lower()
    train_tf, test_tf = _get_transforms(name)
    os.makedirs(data_root, exist_ok=True)

    if name == "kmnist":
        train_ds = torchvision.datasets.KMNIST(
            root=data_root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.KMNIST(
            root=data_root, train=False, download=True, transform=test_tf)
    elif name == "svhn":
        train_ds = torchvision.datasets.SVHN(
            root=data_root, split="train", download=True, transform=train_tf)
        test_ds = torchvision.datasets.SVHN(
            root=data_root, split="test", download=True, transform=test_tf)
    elif name == "stl10":
        train_ds = torchvision.datasets.STL10(
            root=data_root, split="train", download=True, transform=train_tf)
        test_ds = torchvision.datasets.STL10(
            root=data_root, split="test", download=True, transform=test_tf)
    elif name == "eurosat":
        # MODIFICADO: Para evitar modificar el atributo interno .dataset de un Subset, 
        # generamos los índices de corte primero usando una semilla fija.
        full_ds_train = torchvision.datasets.EuroSAT(
            root=data_root, download=True, transform=train_tf)
        full_ds_test = torchvision.datasets.EuroSAT(
            root=data_root, download=False, transform=test_tf)
        
        n = len(full_ds_train)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        n_train = int(0.8 * n)
        
        train_ds = Subset(full_ds_train, indices[:n_train])
        test_ds = Subset(full_ds_test, indices[n_train:])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def get_calibration_loader(
    dataset_name: str,
    n_samples: int = 2048,
    batch_size: int = 256,
    data_root: str = "../data",  # MODIFICADO: Consistencia en la ruta de la carpeta raíz
):
    """
    Return a DataLoader with a random subset of the training set
    (no augmentation). Used for REPAIR calibration and activation matching.
    """
    name = dataset_name.lower()
    _, test_tf = _get_transforms(name)  # no augmentation

    if name == "kmnist":
        full_ds = torchvision.datasets.KMNIST(
            root=data_root, train=True, download=True, transform=test_tf)
    elif name == "svhn":
        full_ds = torchvision.datasets.SVHN(
            root=data_root, split="train", download=True, transform=test_tf)
    elif name == "stl10":
        full_ds = torchvision.datasets.STL10(
            root=data_root, split="train", download=True, transform=test_tf)
    elif name == "eurosat":
        full_ds = torchvision.datasets.EuroSAT(
            root=data_root, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    indices = torch.randperm(
    len(full_ds),
    generator=torch.Generator().manual_seed(42)       # ← add seed
    )[:n_samples]
    subset = Subset(full_ds, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


# ── Helpers ─────────────────────────────────────────────────────────────────

class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clone_model(model):
    """Return a deep copy of a model (weights included)."""
    return copy.deepcopy(model)