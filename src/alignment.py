"""
src/alignment.py — Model alignment methods (Eje A).

Implements three alignment strategies from the LMC literature:

  1. Weight Matching   (Git Re-Basin — Ainsworth et al., 2023)
  2. Activation Matching  (Git Re-Basin)
  3. Procrustes Orthogonal (SVD closed-form, H2)
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


# ═════════════════════════════════════════════════════════════════════════════
# Permutation specification
# ═════════════════════════════════════════════════════════════════════════════
def _get_mlp3_perm_spec():
    """
    Permutation specification for MLP3.
    Returns list of dicts, one per permutable layer, with:
      - 'weight': param name for the weight matrix
      - 'bias': param name for the bias vector
      - 'bn': None (MLP3 has no BN)
    """
    return [
        {"weight": "layer0.weight", "bias": "layer0.bias", "bn": None},
        {"weight": "layer1.weight", "bias": "layer1.bias", "bn": None},
        {"weight": "layer2.weight", "bias": "layer2.bias", "bn": None},
        # layer3 is the output layer — not permuted
    ]


def _get_convbn_perm_spec():
    """
    Permutation specification for SimpleConvBN.

    Conv layers: permuting output channels of conv_i means
      - permuting rows (dim 0) of conv_i.weight
      - permuting cols (dim 1) of conv_{i+1}.weight
      - permuting BN_i params
    """
    return [
        {"weight": "conv0.weight", "bias": "conv0.bias",
         "bn": "bn0", "n_channels": 32},
        {"weight": "conv1.weight", "bias": "conv1.bias",
         "bn": "bn1", "n_channels": 64},
        # Quitamos "spatial" de aquí
        {"weight": "conv2.weight", "bias": "conv2.bias",
         "bn": "bn2", "n_channels": 128}, 
        # Lo movemos aquí, porque fc0 es la que sufre la expansión de canales a bloques en su entrada
        {"weight": "fc0.weight", "bias": "fc0.bias",
         "bn": "bn_fc", "n_channels": 256, "spatial": 16},
    ]


def get_perm_spec(model):
    """Auto-detect model type and return its permutation spec."""
    class_name = model.__class__.__name__
    if class_name == "MLP3":
        return _get_mlp3_perm_spec()
    elif class_name == "SimpleConvBN":
        return _get_convbn_perm_spec()
    else:
        raise ValueError(f"No perm spec for {class_name}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Weight Matching  (Git Re-Basin)
# ═════════════════════════════════════════════════════════════════════════════

def weight_matching(model_a, model_b, max_iter: int = 100):
    """
    Find permutations P_0, P_1, ... that align model_b to model_a
    using coordinate descent on the weight-matching objective.
    """
    spec = get_perm_spec(model_a)
    sd_a = {k: v.cpu() for k, v in model_a.state_dict().items()}
    sd_b = {k: v.cpu() for k, v in model_b.state_dict().items()}

    n_layers = len(spec)
    # Initialize as identity permutations
    perms = [np.arange(sd_a[spec[i]["weight"]].shape[0])
             for i in range(n_layers)]

    for iteration in range(max_iter):
        progress = False

        for layer_idx in range(n_layers):
            w_name = spec[layer_idx]["weight"]
            W_a = sd_a[w_name].float()
            W_b = sd_b[w_name].float()

            # 1. Apply previous layer's permutation to B's input dimension FIRST
            if layer_idx > 0:
                prev_perm = perms[layer_idx - 1]
                spatial = spec[layer_idx].get("spatial", None)
                
                if spatial is not None:
                    # Conv→FC boundary: permute blocks on flattened 2D input columns
                    block_idx = _block_perm_indices(prev_perm, spatial)
                    W_b = W_b[:, block_idx]
                elif W_b.dim() == 4:
                    # Conv→Conv boundary: permute along the input channel axis (dim 1) while still 4D
                    W_b = W_b[:, prev_perm, :, :]
                else:
                    # Standard FC→FC boundary
                    W_b = W_b[:, prev_perm]

            # 2. NOW flatten conv kernels safely after channel sorting is complete
            if W_a.dim() == 4:
                W_a = W_a.flatten(1)
                W_b = W_b.flatten(1)

            # 3. Compute cost matrix: maximize ⟨row_a, row_b⟩
            C = -W_a @ W_b.T  # shape: (n, n)
            row_ind, col_ind = linear_sum_assignment(C.numpy())

            if not np.array_equal(col_ind, perms[layer_idx]):
                perms[layer_idx] = col_ind
                progress = True

        if not progress:
            break

    return perms
    
def _block_perm_indices(channel_perm, spatial_size):
    """
    Given a channel permutation and spatial size (e.g. 4*4=16),
    return the index array that permutes blocks of `spatial_size`
    in the flattened FC input dimension.
    """
    indices = []
    for ch in channel_perm:
        start = ch * spatial_size
        indices.extend(range(start, start + spatial_size))
    return indices


# ═════════════════════════════════════════════════════════════════════════════
# 2. Activation Matching  (Git Re-Basin)
# ═════════════════════════════════════════════════════════════════════════════

def _collect_activations(model, loader, device, max_batches=5):
    """
    Collect post-activation hidden representations at each permutable layer.
    Returns dict: layer_index → Tensor(n_samples, n_units).
    """
    spec = get_perm_spec(model)
    hooks, activations = [], {}

    def _get_hook(idx):
        def hook_fn(module, inp, out):
            #Flatten spatial dims for conv layers
            act = out.detach()
            if act.dim() == 4:
                act = act.mean(dim=(2, 3))  #global average pool per channel
            if idx not in activations:
                activations[idx] = []
            activations[idx].append(act.cpu())
        return hook_fn
    named_modules = dict(model.named_modules())
    for idx, s in enumerate(spec):
        #Find the module that owns this weight
        mod_name = s["weight"].rsplit(".weight", 1)[0]
        mod = named_modules[mod_name]
        hooks.append(mod.register_forward_hook(_get_hook(idx)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            model(x.to(device))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v, dim=0) for k, v in activations.items()}


def activation_matching(model_a, model_b, loader, device):
    """
    Find permutations that align model_b to model_a by matching
    activations using LAP on the cross-correlation matrix:
    """
    acts_a = _collect_activations(model_a, loader, device)
    acts_b = _collect_activations(model_b, loader, device)

    spec = get_perm_spec(model_a)
    perms = []

    for idx in range(len(spec)):
        H_a = acts_a[idx].float()  #(N, d)
        H_b = acts_b[idx].float()
        #Correlation matrix: (d, d)
        corr = H_a.T @ H_b
        cost = -corr.numpy()
        _, col_ind = linear_sum_assignment(cost)
        perms.append(col_ind)
    return perms


# ═════════════════════════════════════════════════════════════════════════════
# 3. Procrustes Orthogonal Alignment  (H2)
# ═════════════════════════════════════════════════════════════════════════════
def procrustes_alignment(model_a, model_b, loader, device):
    """
    Find orthogonal matrices Q^(l) that align activations of model_b
    to model_a via the Procrustes problem:
    """
    acts_a = _collect_activations(model_a, loader, device)
    acts_b = _collect_activations(model_b, loader, device)
    spec = get_perm_spec(model_a)
    transforms = []
    for idx in range(len(spec)):
        H_a = acts_a[idx].float()  #(N, d)
        H_b = acts_b[idx].float()
        #Procrustes: H_B^T H_A = U Σ V^T → Q = U V^T
        M = H_b.T @ H_a  # (d, d)
        U, _, Vt = torch.linalg.svd(M)
        Q = U @ Vt  #orthogonal matrix
        transforms.append(Q)
    return transforms


# ═════════════════════════════════════════════════════════════════════════════
# Apply transformations to state dicts
# ═════════════════════════════════════════════════════════════════════════════

def apply_permutation_to_state_dict(model_b, perms):
    """
    Return a new state_dict with model_b's weights permuted according to
    `perms` (output of weight_matching or activation_matching).

    Does NOT modify model_b in-place.
    """
    spec = get_perm_spec(model_b)
    sd = {k: v.clone().cpu() for k, v in model_b.state_dict().items()}

    for layer_idx, perm in enumerate(perms):
        perm_t = torch.LongTensor(perm)
        s = spec[layer_idx]
        #Permute output dimension of this layer
        w_name = s["weight"]
        sd[w_name] = sd[w_name][perm_t]
        b_name = s["bias"]
        if b_name in sd:
            sd[b_name] = sd[b_name][perm_t]
        #Permute BN parameters
        if s.get("bn") is not None:
            bn_prefix = s["bn"]
            for suffix in [".weight", ".bias", ".running_mean", ".running_var"]:
                key = bn_prefix + suffix
                if key in sd:
                    sd[key] = sd[key][perm_t]
        #Permute input dimension of NEXT layer
        if layer_idx + 1 < len(spec) + 1:
            # ind the next weight layer
            if layer_idx + 1 < len(spec):
                next_w = spec[layer_idx + 1]["weight"]
            else:
                #Last spec layer +1 is the output layer
                class_name = model_b.__class__.__name__
                if class_name == "MLP3":
                    next_w = "layer3.weight"
                elif class_name == "SimpleConvBN":
                    next_w = "fc1.weight"
                else:
                    continue
            if next_w in sd:
                if layer_idx + 1 < len(spec):
                    next_spec = spec[layer_idx + 1]
                    spatial = next_spec.get("spatial", None)
                    if spatial is not None:
                        block_idx = torch.LongTensor(
                            _block_perm_indices(perm, spatial))
                        sd[next_w] = sd[next_w][:, block_idx]
                    else:
                        sd[next_w] = sd[next_w][:, perm_t]
                else:
                    sd[next_w] = sd[next_w][:, perm_t]
    return sd


def apply_transform_to_state_dict(model_b, transforms):
    """
    Apply orthogonal transforms (from procrustes_alignment) to model_b.
    """
    spec = get_perm_spec(model_b)
    sd = {k: v.clone().cpu().float() for k, v in model_b.state_dict().items()}
    for layer_idx, Q in enumerate(transforms):
        s = spec[layer_idx]
        w_name = s["weight"]
        W = sd[w_name]
        is_conv = W.dim() == 4
        if is_conv:
            C_out, C_in, kH, kW = W.shape
            W_flat = W.reshape(C_out, -1)  #(C_out, C_in*kH*kW)
            W_flat = Q @ W_flat  #rotate outputs
            sd[w_name] = W_flat.reshape(C_out, C_in, kH, kW)
        else:
            sd[w_name] = Q @ W  #rotate outputs
        #Bias
        b_name = s["bias"]
        if b_name in sd:
            sd[b_name] = Q @ sd[b_name]
        #BN params
        if s.get("bn") is not None:
            bn_prefix = s["bn"]
            for suffix in [".weight", ".bias"]: 
                key = bn_prefix + suffix
                if key in sd:
                    sd[key] = Q @ sd[key]
            #Keep running statistics as their original un-rotated tensors
            for suffix in [".running_mean", ".running_var"]:
                key = bn_prefix + suffix
                if key in sd:
                    sd[key] = model_b.state_dict()[key].clone().cpu().float()
        #Rotate input dim of next layer
        if layer_idx + 1 < len(spec):
            next_w = spec[layer_idx + 1]["weight"]
            W_next = sd[next_w]
            spatial = spec[layer_idx + 1].get("spatial", None)

            if W_next.dim() == 4:
                C_out, C_in, kH, kW = W_next.shape
                W_flat = W_next.reshape(C_out, C_in, -1)
                #Rotate along C_in: W_next[:, :, :] @ Q^T
                for i in range(C_out):
                    W_flat[i] = (Q.T @ W_flat[i]).clone()
                sd[next_w] = W_flat.reshape(C_out, C_in, kH, kW)
            elif spatial is not None:
                #Conv→FC boundary
                d_out, d_in = W_next.shape
                n_ch = Q.shape[0]
                sp = spatial
                W_blocks = W_next.reshape(d_out, n_ch, sp)
                for i in range(d_out):
                    W_blocks[i] = (Q.T @ W_blocks[i]).clone()
                sd[next_w] = W_blocks.reshape(d_out, d_in)
            else:
                sd[next_w] = W_next @ Q.T
        else:
            #Output layer
            class_name = model_b.__class__.__name__
            if class_name == "MLP3":
                out_w = "layer3.weight"
            elif class_name == "SimpleConvBN":
                out_w = "fc1.weight"
            else:
                continue
            if out_w in sd:
                sd[out_w] = sd[out_w] @ Q.T
    return sd


# ═════════════════════════════════════════════════════════════════════════════
# Cycle-Consistency (Crisostomi et al. / C²M³, 2024)
# ═════════════════════════════════════════════════════════════════════════════
def cycle_consistency_error(model_a, model_b, model_c, method="weight_matching",
                            loader=None, device=None):
    """
    Measure cycle-consistency of a pairwise alignment method.
    """
    if method == "weight_matching":
        perm_ab = weight_matching(model_a, model_b)
        perm_bc = weight_matching(model_b, model_c)
        perm_ac = weight_matching(model_a, model_c)
    elif method == "activation_matching":
        assert loader is not None and device is not None
        perm_ab = activation_matching(model_a, model_b, loader, device)
        perm_bc = activation_matching(model_b, model_c, loader, device)
        perm_ac = activation_matching(model_a, model_c, loader, device)
    else:
        raise ValueError(f"Unknown method: {method}")
    #Direct: align C to A
    sd_c_direct = apply_permutation_to_state_dict(model_c, perm_ac)  
    #We need a workaround: apply perm_bc to C, then perm_ab to the result
    sd_c_to_b = apply_permutation_to_state_dict(model_c, perm_bc)
    model_c_temp = copy.deepcopy(model_c)
    model_c_temp.load_state_dict(sd_c_to_b)
    sd_c_indirect = apply_permutation_to_state_dict(model_c_temp, perm_ab)
    #Compute relative L2 distance
    diff_sq = 0.0
    norm_sq = 0.0
    for key in sd_c_direct:
        if sd_c_direct[key].is_floating_point():
            d = (sd_c_direct[key].float() - sd_c_indirect[key].float())
            diff_sq += (d ** 2).sum().item()
            norm_sq += (sd_c_direct[key].float() ** 2).sum().item()

    error = (diff_sq ** 0.5) / max(norm_sq ** 0.5, 1e-12)
    return error
