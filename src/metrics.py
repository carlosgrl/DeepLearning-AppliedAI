"""
src/metrics.py — Barrier computation, CKA, and model interpolation.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
# ═════════════════════════════════════════════════════════════════════════════
# Model interpolation
# ═════════════════════════════════════════════════════════════════════════════

def interpolate_state_dicts(sd_a, sd_b, alpha):
    """
    Linearly interpolate two state dicts
    """
    sd_interp = {}
    for key in sd_a:
        if sd_a[key].dtype in (torch.long, torch.int32, torch.int64):
            sd_interp[key] = sd_a[key]  #eg num_batches_tracked
        else:
            sd_interp[key] = alpha * sd_a[key].float() + \
                             (1 - alpha) * sd_b[key].float()
    return sd_interp


def make_interpolated_model(model_class, model_kwargs, sd_a, sd_b,
                            alpha, device):
    """Create a new model with interpolated weights."""
    sd = interpolate_state_dicts(sd_a, sd_b, alpha)
    model = model_class(**model_kwargs)
    model.load_state_dict(sd)
    return model.to(device)


# ═════════════════════════════════════════════════════════════════════════════
# Linear loss barrier
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _eval_loss(model, loader, device, criterion=None):
    """Compute average loss."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


@torch.no_grad()
def _eval_acc(model, loader, device):
    """Compute accuracy."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def compute_barrier(
    model_class,
    model_kwargs,
    sd_a,
    sd_b,
    loader,
    device,
    n_steps: int = 21,
    return_curve: bool = False,
):
    """
    Compute the linear loss barrier B(θ_A, θ_B)
    """
    criterion = nn.CrossEntropyLoss()
    alphas = np.linspace(0, 1, n_steps)
    losses, accs = [], []

    model = model_class(**model_kwargs).to(device)
    for alpha in alphas:
        sd_interp = interpolate_state_dicts(sd_a, sd_b, alpha)
        model.load_state_dict(sd_interp)
        model.eval()
        loss = _eval_loss(model, loader, device, criterion)
        acc = _eval_acc(model, loader, device)
        losses.append(loss)
        accs.append(acc)

    losses = np.array(losses)
    accs = np.array(accs)

    #Barrier = max interpolated loss − average endpoint loss
    endpoint_avg = (losses[0] + losses[-1]) / 2
    barrier = losses.max() - endpoint_avg

    if return_curve:
        return barrier, {
            "alphas": alphas,
            "losses": losses,
            "accs": accs,
        }
    return barrier

def compute_error_barrier(barrier_curve):
    """Compute error barrier from a curve dict returned by compute_barrier."""
    accs = barrier_curve["accs"]
    errors = 1 - accs
    endpoint_avg = (errors[0] + errors[-1]) / 2
    return errors.max() - endpoint_avg

# ═════════════════════════════════════════════════════════════════════════════
# CKA — Centered Kernel Alignment (Kornblith et al., 2019)
# ═════════════════════════════════════════════════════════════════════════════
def cka_linear(X, Y):
    """
    Compute linear CKA between two representation matrices
    """
    X = X.float()
    Y = Y.float()

    #Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (Y.T @ X).norm("fro") ** 2
    hsic_xx = (X.T @ X).norm("fro")
    hsic_yy = (Y.T @ Y).norm("fro")

    denom = hsic_xx * hsic_yy
    if denom < 1e-12:
        return 0.0
    return (hsic_xy / denom).item()


def _collect_representations(model, loader, device, max_batches=5):
    """
    Collect hidden representations at every Linear/Conv layer.
    Returns dict: layer_name → Tensor(n_samples, n_features).
    """
    reps = {}
    hooks = []

    def _hook(name):
        def fn(module, inp, out):
            act = out.detach()
            if act.dim() == 4:
                act = act.mean(dim=(2, 3))  #GAP per channel
            if name not in reps:
                reps[name] = []
            reps[name].append(act.cpu())
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            hooks.append(mod.register_forward_hook(_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    return {k: torch.cat(v, dim=0) for k, v in reps.items()}


def compute_cka_matrix(model_a, model_b, loader, device, max_batches=5):
    """
    Compute the full layer-by-layer CKA matrix between two models
    """
    reps_a = _collect_representations(model_a, loader, device, max_batches)
    reps_b = _collect_representations(model_b, loader, device, max_batches)

    names_a = list(reps_a.keys())
    names_b = list(reps_b.keys())
    n_a, n_b = len(names_a), len(names_b)

    cka_mat = np.zeros((n_a, n_b))
    for i, na in enumerate(names_a):
        for j, nb in enumerate(names_b):
            cka_mat[i, j] = cka_linear(reps_a[na], reps_b[nb])

    return cka_mat, names_a, names_b
