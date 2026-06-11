"""
src/training.py — Training loops, spawning, and checkpointing.
"""

import os
import copy
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .utils import AverageMeter, set_seed


# ── Single epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_meter.avg, correct / total


# ── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_meter.avg, correct / total


# ── Full training loop ──────────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    test_loader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device=None,
    save_dir: str = None,
    tag: str = "model",
    verbose: bool = True,
):
    """
    Full training with Adam, cosine annealing, and optional checkpointing.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {tag}")

    for epoch in iterator:
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if verbose:
            iterator.set_postfix(
                tr_acc=f"{tr_acc:.3f}", te_acc=f"{te_acc:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}")

    # Save final checkpoint
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{tag}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "history": history,
        }, path)
        if verbose:
            print(f"  → Saved {path}")

    return history


# ── Spawning (shared early training) ────────────────────────────────────────
def train_spawned_pair(
    model_class,
    model_kwargs: dict,
    train_loader,
    test_loader,
    total_epochs: int = 30,
    spawn_epoch: int = 5,
    seed_common: int = 0,
    seed_a: int = 1,
    seed_b: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device=None,
    save_dir: str = None,
    verbose: bool = True,
):
    """
    Train two models that share the first `spawn_epoch` epochs of training,
    then diverge with different random seeds.
    """
    from .utils import get_device  #CORRECCIÓN: Se elimina get_dataloaders redundante
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: shared training ──
    set_seed(seed_common)
    model_common = model_class(**model_kwargs).to(device)
    opt_common = torch.optim.Adam(model_common.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    if verbose:
        print(f"Phase 1: shared training for {spawn_epoch} epochs "
              f"(seed={seed_common})")

    for ep in range(spawn_epoch):
        train_one_epoch(model_common, train_loader, opt_common,
                        criterion, device)

    # ── Phase 2: diverge ──
    model_a = copy.deepcopy(model_common)
    model_b = copy.deepcopy(model_common)
    remaining = total_epochs - spawn_epoch

    if verbose:
        print(f"Phase 2: independent training for {remaining} epochs")

    # ── Train Branch A — completely isolated seed ──
    set_seed(seed_a)
    g_a = torch.Generator()
    g_a.manual_seed(seed_a)
    
    # Reconstrucción limpia y aislada del cargador de datos para la Rama A
    loader_a = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        generator=g_a,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )
    hist_a = train_model(
        model_a, loader_a, test_loader, epochs=remaining,
        lr=lr, weight_decay=weight_decay, device=device,
        save_dir=save_dir, tag=f"spawned_k{spawn_epoch}_A",
        verbose=verbose)

    # ── Train Branch B — completely isolated seed ──
    set_seed(seed_b)
    g_b = torch.Generator()
    g_b.manual_seed(seed_b)
    
    # Reconstrucción limpia y aislada del cargador de datos para la Rama B
    loader_b = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        generator=g_b,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )
    hist_b = train_model(
        model_b, loader_b, test_loader, epochs=remaining,
        lr=lr, weight_decay=weight_decay, device=device,
        save_dir=save_dir, tag=f"spawned_k{spawn_epoch}_B",
        verbose=verbose)

    return model_a, model_b, hist_a, hist_b


# ── Multi-seed experiment runner ────────────────────────────────────────────

def run_multi_seed(experiment_fn, seeds, **kwargs):
    """
    Run an experiment function across multiple seeds and aggregate results.
    """
    import numpy as np

    all_results = []
    for seed in seeds:
        set_seed(seed)
        result = experiment_fn(seed=seed, **kwargs)
        all_results.append(result)
        print(f"  Seed {seed}: {result}")

    # Aggregate
    aggregated = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        aggregated[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],
        }

    print(f"\n  Aggregated ({len(seeds)} seeds):")
    for key, v in aggregated.items():
        print(f"    {key}: {v['mean']:.4f} ± {v['std']:.4f}")

    return aggregated
