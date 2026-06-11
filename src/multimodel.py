"""
src/multimodel.py — Multi-model merging for n>2 (Eje D / H4).

Methods:
  1. Naive arithmetic mean
  2. Pairwise alignment to anchor + averaging
  3. Iterative pairwise merging
  4. TIES-Merging baseline
"""

import copy
import torch
import torch.nn as nn
import numpy as np

from .alignment import weight_matching, apply_permutation_to_state_dict
from .metrics import interpolate_state_dicts, _eval_loss, _eval_acc


def train_n_models(model_class, model_kwargs, train_loader, test_loader,
                   n_models, seeds, epochs=30, lr=1e-3, device=None,
                   save_dir=None, tag_prefix="model"):
    """
    Train n models with different seeds. Returns list of state_dicts.
    """
    from .utils import set_seed, get_device
    from .training import train_model

    if device is None:
        device = get_device()

    state_dicts = []
    histories = []

    for i, seed in enumerate(seeds[:n_models]):
        set_seed(seed)
        model = model_class(**model_kwargs).to(device)
        hist = train_model(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device,
            save_dir=save_dir, tag=f"{tag_prefix}_{i}_s{seed}",
            verbose=True,
        )
        state_dicts.append({k: v.cpu() for k, v in model.state_dict().items()})
        histories.append(hist)

    return state_dicts, histories


# ── 1. Naive arithmetic mean ────────────────────────────────────────────────

def naive_mean_merge(state_dicts):
    """Simple arithmetic mean of n state dicts."""
    n = len(state_dicts)
    merged = {}
    for key in state_dicts[0]:
        if state_dicts[0][key].dtype in (torch.long, torch.int32, torch.int64):
            merged[key] = state_dicts[0][key]
        else:
            merged[key] = sum(sd[key].float() for sd in state_dicts) / n
    return merged


# ── 2. Anchor alignment + averaging ─────────────────────────────────────────

def anchor_aligned_merge(model_class, model_kwargs, state_dicts, device=None):
    """
    Align all models to the first one (anchor), then average.
    Uses weight matching for pairwise alignment.
    """
    from .utils import get_device
    if device is None:
        device = get_device()

    anchor_sd = state_dicts[0]
    anchor_model = model_class(**model_kwargs)
    anchor_model.load_state_dict(anchor_sd)

    aligned_sds = [anchor_sd]

    for i in range(1, len(state_dicts)):
        other = model_class(**model_kwargs)
        other.load_state_dict(state_dicts[i])

        perms = weight_matching(anchor_model, other)
        aligned_sd = apply_permutation_to_state_dict(other, perms)
        aligned_sds.append(aligned_sd)

    return naive_mean_merge(aligned_sds)


# ── 3. Iterative pairwise merging ───────────────────────────────────────────

def iterative_pairwise_merge(model_class, model_kwargs, state_dicts,
                             device=None):
    """
    Merge models pairwise in sequence: ((θ_1 ⊕ θ_2) ⊕ θ_3) ⊕ ...
    Each merge: align second to first via WM, then average.
    """
    from .utils import get_device
    if device is None:
        device = get_device()

    current_sd = state_dicts[0]

    for i in range(1, len(state_dicts)):
        m_curr = model_class(**model_kwargs)
        m_curr.load_state_dict(current_sd)
        m_next = model_class(**model_kwargs)
        m_next.load_state_dict(state_dicts[i])

        perms = weight_matching(m_curr, m_next)
        aligned_sd = apply_permutation_to_state_dict(m_next, perms)

        current_sd = interpolate_state_dicts(current_sd, aligned_sd, alpha=0.5)

    return current_sd


# ── 4. TIES-Merging (Yadav et al., 2023) ────────────────────────────────────

def ties_merge(state_dicts, base_sd=None, k_fraction=0.2):
    """
    TIES-Merging: Trim, Elect Sign, Merge.
    """
    n = len(state_dicts)

    if base_sd is None:
        base_sd = naive_mean_merge(state_dicts)

    #Task vectors
    task_vectors = []
    for sd in state_dicts:
        tv = {}
        for key in sd:
            if sd[key].dtype in (torch.long, torch.int32, torch.int64):
                continue
            tv[key] = sd[key].float() - base_sd[key].float()
        task_vectors.append(tv)

    merged = {}
    for key in base_sd:
        if base_sd[key].dtype in (torch.long, torch.int32, torch.int64):
            merged[key] = base_sd[key]
            continue

        #Stack task vectors for this param
        tvs = torch.stack([tv[key] for tv in task_vectors])  # (n, *shape)

        #Trim: zero out smallest magnitudes
        flat = tvs.abs().reshape(n, -1)
        threshold = torch.quantile(flat, 1 - k_fraction, dim=1, keepdim=True)
        threshold = threshold.reshape(n, *([1] * (tvs.dim() - 1)))
        mask = tvs.abs() >= threshold
        tvs_trimmed = tvs * mask.float()

        #Elect sign: majority vote
        signs = tvs_trimmed.sign()
        sign_sum = signs.sum(dim=0)
        elected_sign = sign_sum.sign()
        #Break ties by defaulting to positive:
        elected_sign[elected_sign == 0] = 1.0

        #Disjoint merge: average only agreeing params
        agree = (signs == elected_sign.unsqueeze(0))
        tvs_filtered = tvs_trimmed * agree.float()
        counts = agree.float().sum(dim=0).clamp(min=1)
        merged_tv = tvs_filtered.sum(dim=0) / counts

        merged[key] = base_sd[key].float() + merged_tv

    return merged


# ── 5. SLERP barycenter (spherical mean for n>2) ────────────────────────────

def _flatten_sd(sd):
    """Flatten float params of a state dict into a 1D tensor."""
    parts = []
    for key in sorted(sd.keys()):
        if sd[key].is_floating_point():
            parts.append(sd[key].float().flatten())
    return torch.cat(parts)


def _unflatten_sd(vec, template_sd):
    """Reconstruct state dict from flat vector."""
    sd = {}
    offset = 0
    for key in sorted(template_sd.keys()):
        if template_sd[key].is_floating_point():
            n = template_sd[key].numel()
            sd[key] = vec[offset:offset + n].reshape(template_sd[key].shape)
            offset += n
        else:
            sd[key] = template_sd[key]
    return sd


def _slerp(v0, v1, t):
    """Spherical linear interpolation between two unit vectors."""
    dot = torch.dot(v0, v1).clamp(-1, 1)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return (1 - t) * v0 + t * v1
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * v0 + \
           (torch.sin(t * omega) / sin_omega) * v1


def slerp_barycenter_merge(state_dicts, max_iter=50, tol=1e-6):
    """
    Compute the spherical barycenter (Fréchet mean on S^{d-1})
    of n parameter vectors.
    """
    template = state_dicts[0]
    vecs = [_flatten_sd(sd) for sd in state_dicts]
    norms = [v.norm() for v in vecs]
    avg_norm = sum(norms) / len(norms)

    # Normalize to unit sphere
    unit_vecs = [v / v.norm() for v in vecs]

    # Initialize at arithmetic mean (projected)
    mu = sum(unit_vecs) / len(unit_vecs)
    mu = mu / mu.norm()

    for it in range(max_iter):
        # Log map: tangent vectors at mu
        tangents = []
        for v in unit_vecs:
            dot = torch.dot(mu, v).clamp(-1, 1)
            angle = torch.acos(dot)
            if angle.abs() < 1e-8:
                tangents.append(torch.zeros_like(mu))
            else:
                direction = v - dot * mu
                direction = direction / direction.norm().clamp(min=1e-12)
                tangents.append(angle * direction)

        # Average tangent
        avg_tangent = sum(tangents) / len(tangents)
        step_size = avg_tangent.norm()

        if step_size < tol:
            break

        # Exp map: move along geodesic
        direction = avg_tangent / step_size.clamp(min=1e-12)
        mu = mu * torch.cos(step_size) + direction * torch.sin(step_size)
        mu = mu / mu.norm()

    # Scale back to original norm
    result_vec = mu * avg_norm
    return _unflatten_sd(result_vec, template)


# ── Multi-model barrier ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_multimodel_barrier(model_class, model_kwargs, merged_sd,
                               individual_sds, loader, device):
    """
    Compute B_n
    """
    criterion = nn.CrossEntropyLoss()

    # Merged loss
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(merged_sd)
    merged_loss = _eval_loss(model, loader, device, criterion)
    merged_acc = _eval_acc(model, loader, device)

    # Individual losses
    ind_losses = []
    ind_accs = []
    for sd in individual_sds:
        model.load_state_dict(sd)
        ind_losses.append(_eval_loss(model, loader, device, criterion))
        ind_accs.append(_eval_acc(model, loader, device))

    avg_ind_loss = np.mean(ind_losses)
    barrier = merged_loss - avg_ind_loss

    return {
        "barrier": barrier,
        "merged_loss": merged_loss,
        "merged_acc": merged_acc,
        "individual_losses": ind_losses,
        "individual_accs": ind_accs,
    }
