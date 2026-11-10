"""Original-feature predictive reliance and model-output attributions.

Methods answer different questions: IG is a path-based attribution, permutation
importance measures predictive loss degradation, and neither is causal.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import log_loss


def model_output(model, x: torch.Tensor, output: str = 'class0_margin'):
    y = model(x)
    if output == 'class0_margin':
        if y.shape[1] != 2:
            raise ValueError('class0_margin requires a binary two-logit classifier')
        return y[:, 0] - y[:, 1]
    if output == 'regression':
        return y[:, 0]
    raise ValueError(output)


def integrated_gradients(model, inputs, baselines, *, steps=1024,
                         output='class0_margin', chunk_size=32):
    """Trapezoidal integrated gradients and completeness residual.

    Uses the same baseline for each sample when baselines shape is [d]. This
    is a numerical approximation. Verify completeness at increased steps.
    """
    if steps < 2:
        raise ValueError('steps must be >=2')
    x = torch.as_tensor(inputs, dtype=torch.float32)
    b = torch.as_tensor(baselines, dtype=torch.float32)
    if b.ndim == 1:
        b = b.unsqueeze(0).expand_as(x)
    if x.ndim != 2 or b.shape != x.shape:
        raise ValueError('inputs and baselines must be [B,d]')
    device = next(model.parameters()).device
    x, b = x.to(device), b.to(device)
    delta = x - b
    model.eval()
    accumulated = torch.zeros_like(x)
    for start in range(0, steps + 1, chunk_size):
        ids = torch.arange(start, min(steps + 1, start + chunk_size), device=device)
        alpha = ids.to(x.dtype) / steps
        points = b.unsqueeze(0) + alpha[:, None, None] * delta.unsqueeze(0)
        flat = points.reshape(-1, x.shape[1]).detach().requires_grad_(True)
        with torch.enable_grad():
            out = model_output(model, flat, output)
            grad = torch.autograd.grad(out.sum(), flat)[0]
        grad = grad.detach().reshape(len(ids), x.shape[0], x.shape[1])
        weights = torch.ones(len(ids), device=device, dtype=x.dtype)
        weights[ids == 0] = .5
        weights[ids == steps] = .5
        accumulated += (grad * weights[:, None, None]).sum(dim=0)
    attribution = delta * accumulated / steps
    with torch.no_grad():
        expected_change = model_output(model, x, output) - model_output(model, b, output)
    residual = expected_change - attribution.sum(dim=1)
    return attribution.detach().cpu().numpy(), residual.detach().cpu().numpy()


def permutation_nll(model, X, y, *, repetitions=10, seed=42, batch_size=256):
    """Held-out NLL degradation for each encoded input column."""
    from scipy.special import softmax
    model.eval()
    device = next(model.parameters()).device
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    def probs(arr):
        all_logits = []
        with torch.no_grad():
            for start in range(0, len(arr), batch_size):
                t = torch.as_tensor(arr[start:start+batch_size], device=device)
                all_logits.append(model(t).cpu().numpy())
        return softmax(np.concatenate(all_logits, axis=0), axis=1)
    base = log_loss(y, probs(X), labels=list(range(model.config.output_dim)))
    rng = np.random.default_rng(seed)
    matrix = np.empty((X.shape[1], repetitions))
    for j in range(X.shape[1]):
        for k in range(repetitions):
            corrupted = X.copy()
            corrupted[:, j] = X[rng.permutation(len(X)), j]
            matrix[j, k] = log_loss(y, probs(corrupted), labels=list(range(model.config.output_dim))) - base
    return matrix.mean(axis=1), matrix.std(axis=1, ddof=1) if repetitions > 1 else np.zeros(X.shape[1])
