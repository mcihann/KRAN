"""Original-input interventions for the strictly additive KRAN model.

The direct/downstream split is an algebraic decomposition of the trained model,
NOT a decomposition of causal direct and indirect effects.
"""
from __future__ import annotations

import numpy as np
import torch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def original_feature_responses(model, references: np.ndarray, feature_index: int,
                               grid: np.ndarray, batch_size: int = 256):
    """Compute reference-conditioned responses to replacing input feature j.

    Args:
      references: [R,d] TRAIN-derived scaled feature vectors in [0,1].
      feature_index: index in current KR-MonoMap order.
      grid: candidate scaled feature values; may be non-uniform.

    Returns:
      dict of [R,len(grid)] arrays (direct, downstream, total) for regression
      or binary class-0 minus class-1 log-odds, plus probability change in the
      binary case. Reference baseline is its *own original* feature value.
    """
    if getattr(model.config, 'head_type', None) != 'additive':
        raise TypeError('Exact direct/downstream attribution requires additive head')
    if model.mono_map is None:
        raise TypeError('Requires a learned triangular transport')
    if model.config.task == 'classification' and model.config.output_dim != 2:
        raise ValueError('Binary log-odds response routine requires exactly two class logits')
    references = np.asarray(references, dtype=np.float32)
    grid = np.asarray(grid, dtype=np.float32).reshape(-1)
    if references.ndim != 2 or references.shape[1] != model.config.input_dim:
        raise ValueError('references must have shape [R,input_dim]')
    if not 0 <= feature_index < model.config.input_dim:
        raise IndexError('feature_index out of range')
    if np.any((references < 0) | (references > 1)) or np.any((grid < 0) | (grid > 1)):
        raise ValueError('Inputs must be scaled to [0,1]')
    model.eval()
    device = next(model.parameters()).device
    R, d = references.shape
    N = len(grid)
    repeats = np.repeat(references, N, axis=0)
    repeats[:, feature_index] = np.tile(grid, R)

    def predict_effects(data):
        outputs = []
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                xt = torch.as_tensor(data[start:start+batch_size], device=device)
                z, _ = model.mono_map(xt)
                e = model.head.feature_effects(z).cpu().numpy()  # [batch,d,out]
                outputs.append(e[..., 0] - e[..., 1] if model.config.task == 'classification' else e[..., 0])
        return np.concatenate(outputs, axis=0)

    baseline = predict_effects(references)   # [R,d]
    intervention = predict_effects(repeats).reshape(R, N, d)
    delta = intervention - baseline[:, None, :]
    direct = delta[:, :, feature_index]
    downstream = delta[:, :, feature_index + 1:].sum(axis=-1)
    total = delta.sum(axis=-1)
    out = {'direct': direct, 'downstream': downstream, 'total': total,
           'decomposition_max_error': float(np.max(np.abs(total - direct - downstream)))}
    if model.config.task == 'classification':
        # bias contribution to class0 minus class1 margin
        bias = (model.head.bias[0] - model.head.bias[1]).detach().cpu().item()
        base_margin = baseline.sum(axis=1) + bias
        intervened_margin = intervention.sum(axis=2) + bias
        out['probability_change_class0'] = _sigmoid(intervened_margin) - _sigmoid(base_margin)[:, None]
    return out


def summarize_response(out: dict, grid: np.ndarray):
    """Means and descriptive 5th/95th percentiles across reference profiles."""
    result = {'scaled_value': np.asarray(grid)}
    for name in ('direct', 'downstream', 'total', 'probability_change_class0'):
        if name not in out:
            continue
        arr = out[name]
        result[f'{name}_mean'] = arr.mean(axis=0)
        result[f'{name}_q05'] = np.percentile(arr, 5, axis=0)
        result[f'{name}_q95'] = np.percentile(arr, 95, axis=0)
    return result
