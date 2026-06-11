"""
src/repair.py — REPAIR and variance-collapse analysis (Eje B).
"""

import copy
import torch
import torch.nn as nn
from .utils import AverageMeter


# ═════════════════════════════════════════════════════════════════════════════
# Collect per-layer activation statistics
# ═════════════════════════════════════════════════════════════════════════════

def _get_hookable_layers(model):
    """
    Return a list of (name, module) for layers after which we want
    to measure activation statistics.
    
    FIX APPLIED: Strictly hook Conv2d and Linear modules. 
    This prevents the "Shared ReLU Tensor Crash" where reused activation 
    functions mix tensors of different channel sizes (e.g., 32 and 64) 
    into the same hook list.
    """
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            layers.append((name, mod))
    return layers


def collect_activation_stats(model, loader, device, max_batches=10):
    """
    Forward calibration data through the model and collect per-layer
    activation statistics (mean and std per unit/channel).

    Returns
    -------
    stats : dict[str, dict] with keys 'mean' and 'std' per layer name
    """
    layers = _get_hookable_layers(model)
    raw = {}
    hooks = []

    def _hook(name):
        def fn(module, inp, out):
            act = out.detach()
            if act.dim() == 4:
                # Conv: compute stats per channel (dim 1)
                act = act.permute(0, 2, 3, 1).reshape(-1, act.size(1))
            else:
                act = act.reshape(-1, act.size(-1))
            if name not in raw:
                raw[name] = []
            raw[name].append(act.cpu())
        return fn

    for name, mod in layers:
        hooks.append(mod.register_forward_hook(_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    stats = {}
    for name, acts_list in raw.items():
        all_acts = torch.cat(acts_list, dim=0).float()
        stats[name] = {
            "mean": all_acts.mean(dim=0),
            "std": all_acts.std(dim=0).clamp(min=1e-8),
        }
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# Variance collapse ratio (diagnostic metric)
# ═════════════════════════════════════════════════════════════════════════════

def variance_collapse_ratio(model_merged, model_a, model_b,
                            loader, device, max_batches=10):
    """
    Compute the per-layer variance collapse ratio
    """
    stats_m = collect_activation_stats(model_merged, loader, device,
                                       max_batches)
    stats_a = collect_activation_stats(model_a, loader, device, max_batches)
    stats_b = collect_activation_stats(model_b, loader, device, max_batches)

    ratios = {}
    for name in stats_m:
        if name in stats_a and name in stats_b:
            sigma_expected = (stats_a[name]["std"] + stats_b[name]["std"]) / 2
            sigma_merged = stats_m[name]["std"]
            ratios[name] = (sigma_merged / sigma_expected.clamp(min=1e-8))

    return ratios


# ═════════════════════════════════════════════════════════════════════════════
# REPAIR — Full (Jordan et al., 2023)
# ═════════════════════════════════════════════════════════════════════════════

class _RescaleHook:
    """Forward hook that rescales activations to target statistics."""

    def __init__(self, target_mean, target_std, actual_mean, actual_std):
        self.target_mean = target_mean
        self.target_std = target_std
        self.actual_mean = actual_mean
        self.actual_std = actual_std

    def __call__(self, module, inp, out):
        if out.dim() == 4:
            # Conv: (N, C, H, W) — reshape stats to (1, C, 1, 1)
            m = self.actual_mean.to(out.device).view(1, -1, 1, 1)
            s = self.actual_std.to(out.device).view(1, -1, 1, 1)
            tm = self.target_mean.to(out.device).view(1, -1, 1, 1)
            ts = self.target_std.to(out.device).view(1, -1, 1, 1)
        else:
            m = self.actual_mean.to(out.device)
            s = self.actual_std.to(out.device)
            tm = self.target_mean.to(out.device)
            ts = self.target_std.to(out.device)

        return (out - m) / s.clamp(min=1e-8) * ts + tm


def repair_full(model_merged, model_a, model_b, alpha,
                calib_loader, device):
    """
    Full REPAIR: insert parameter-free rescaling layers that restore
    the expected interpolated statistics at every hidden layer.
    """
    model_repaired = copy.deepcopy(model_merged).to(device)

    # Collect stats from parents
    stats_a = collect_activation_stats(model_a, calib_loader, device)
    stats_b = collect_activation_stats(model_b, calib_loader, device)

    # Collect actual stats of merged model
    stats_m = collect_activation_stats(model_repaired, calib_loader, device)

    # Install rescaling hooks
    layers = _get_hookable_layers(model_repaired)
    hooks = []

    for name, mod in layers:
        if name not in stats_a or name not in stats_b or name not in stats_m:
            continue

        # Target: interpolated statistics
        target_mean = alpha * stats_a[name]["mean"] + \
                      (1 - alpha) * stats_b[name]["mean"]
        target_std = (alpha * stats_a[name]["std"] + \
                      (1 - alpha) * stats_b[name]["std"])

        actual_mean = stats_m[name]["mean"]
        actual_std = stats_m[name]["std"]

        hook = _RescaleHook(target_mean, target_std, actual_mean, actual_std)
        hooks.append(mod.register_forward_hook(hook))

    # Store hooks for cleanup
    model_repaired._repair_hooks = hooks
    return model_repaired


# ═════════════════════════════════════════════════════════════════════════════
# REPAIR — BN recalibration only
# ═════════════════════════════════════════════════════════════════════════════

def repair_bn_recalibration(model_merged, calib_loader, device):
    """
    Re-estimate BatchNorm running statistics by forwarding
    calibration data through the merged model.
    """
    model_recalib = copy.deepcopy(model_merged).to(device)

    # Reset BN stats
    for mod in model_recalib.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.reset_running_stats()
            mod.momentum = None  # use cumulative moving average

    # Forward pass in train mode to accumulate stats
    model_recalib.train()
    with torch.no_grad():
        for x, _ in calib_loader:
            model_recalib(x.to(device))

    # Restore default momentum after calibration
    for mod in model_recalib.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.momentum = 0.1                        

    model_recalib.eval()
    return model_recalib


# ═════════════════════════════════════════════════════════════════════════════
# REPAIR — Reset & retrain BN affine parameters (γ, β)
# ═════════════════════════════════════════════════════════════════════════════

def repair_reset_retrain_bn(model_merged, calib_loader, device,
                            lr=0.01, epochs=3):
    """
    Reset BN running stats, then retrain the BN affine parameter while keeping all other weights frozen.
    """
    model_ret = copy.deepcopy(model_merged).to(device)

    # Freeze everything
    for p in model_ret.parameters():
        p.requires_grad = False

    # Unfreeze and reset BN params
    bn_params = []
    for mod in model_ret.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.reset_running_stats()
            mod.momentum = None
            if mod.weight is not None:
                mod.weight.requires_grad = True
                mod.weight.data.fill_(1.0)
                bn_params.append(mod.weight)
            if mod.bias is not None:
                mod.bias.requires_grad = True
                mod.bias.data.fill_(0.0)
                bn_params.append(mod.bias)

    if not bn_params:
        # No BN layers — just return
        return model_ret

    optimizer = torch.optim.SGD(bn_params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model_ret.train()
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model_ret(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_ret.eval()
    return model_ret


# ═════════════════════════════════════════════════════════════════════════════
# REPAIR — Layerwise (apply only to selected layers)
# ═════════════════════════════════════════════════════════════════════════════

def repair_layerwise(model_merged, model_a, model_b, alpha,
                     calib_loader, device, layer_names=None):
    """
    Apply REPAIR rescaling only to the specified layers.
    Useful for ablation: which layers benefit most from repair?

    If layer_names is None, applies to all layers (= repair_full).
    """
    model_repaired = copy.deepcopy(model_merged).to(device)

    stats_a = collect_activation_stats(model_a, calib_loader, device)
    stats_b = collect_activation_stats(model_b, calib_loader, device)
    stats_m = collect_activation_stats(model_repaired, calib_loader, device)

    layers = _get_hookable_layers(model_repaired)
    hooks = []

    for name, mod in layers:
        if layer_names is not None and name not in layer_names:
            continue
        if name not in stats_a or name not in stats_b or name not in stats_m:
            continue

        target_mean = alpha * stats_a[name]["mean"] + \
                      (1 - alpha) * stats_b[name]["mean"]
        target_std = alpha * stats_a[name]["std"] + \
                     (1 - alpha) * stats_b[name]["std"]

        hook = _RescaleHook(target_mean, target_std,
                            stats_m[name]["mean"], stats_m[name]["std"])
        hooks.append(mod.register_forward_hook(hook))

    model_repaired._repair_hooks = hooks
    return model_repaired


# ═════════════════════════════════════════════════════════════════════════════
# Variance collapse vs α sweep (§7.4)
# ═════════════════════════════════════════════════════════════════════════════

def variance_collapse_vs_alpha(model_class, model_kwargs, sd_a, sd_b,
                                loader, device,
                                alphas=None, max_batches=10):
    """
    Compute per-layer variance collapse ratio r^(l) for multiple α values.
    """
    from .metrics import interpolate_state_dicts

    if alphas is None:
        alphas = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Collect parent stats once
    model_a = model_class(**model_kwargs).to(device)
    model_a.load_state_dict(sd_a)
    stats_a = collect_activation_stats(model_a, loader, device, max_batches)

    model_b = model_class(**model_kwargs).to(device)
    model_b.load_state_dict(sd_b)
    stats_b = collect_activation_stats(model_b, loader, device, max_batches)

    results = {}
    for alpha in alphas:
        sd_interp = interpolate_state_dicts(sd_a, sd_b, alpha)
        model_m = model_class(**model_kwargs).to(device)
        model_m.load_state_dict(sd_interp)
        stats_m = collect_activation_stats(model_m, loader, device, max_batches)

        ratios = {}
        for name in stats_m:
            if name in stats_a and name in stats_b:
                sigma_expected = alpha * stats_a[name]["std"] + \
                                 (1 - alpha) * stats_b[name]["std"]
                sigma_merged = stats_m[name]["std"]
                ratios[name] = (sigma_merged /
                                sigma_expected.clamp(min=1e-8))
        results[alpha] = ratios

    return results
