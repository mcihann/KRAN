"""Synthetic scalability evaluation with isolated CUDA peak-memory measurements.

Each (dimension, quadrature resolution) configuration launches a fresh process.
No GPU? Peak CUDA fields are empty, not fabricated from CPU memory.
Results are measured at runtime; CUDA allocator peak-memory values require a GPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from kran.metrics import regression_metrics
from kran.model import KRAN, KRANConfig
from kran.train import loaders, predict, set_seed, train_model

DIMS = (25, 50, 100, 200, 500)
GRIDS = (8, 16, 32)


def synthetic_data(dim, num_samples=5000):
    # Maintain deterministic sequential RNG consumption across dimensions.
    rng = np.random.RandomState(42)
    for d in DIMS:
        x = rng.rand(num_samples, d).astype(np.float32)
        y = (np.sin(2*np.pi*x[:,0]) + x[:,1]**2 + x[:,2]*x[:,3]
             + np.log1p(x[:,4]) + rng.normal(0,.05,num_samples)).astype(np.float32)[:,None]
        if d == dim:
            return train_test_split(x, y, test_size=.2, random_state=42)
    raise ValueError(dim)


def run_one(dim, grid, device):
    set_seed(42)
    Xtr, Xte, ytr, yte = synthetic_data(dim)
    tr, te = loaders(Xtr, ytr, Xte, yte, 256, 'regression')
    if device == 'cuda':
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = KRAN(KRANConfig(dim, 1, task='regression', mono_num_quad_points=grid))
    _, train_time = train_model(model, tr, 'regression', 5, lr=1e-3,
                                weight_decay=1e-4, device=device)
    yt, yp, infer_time = predict(model, te, 'regression', device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / (1024**2)
        reserved = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        allocated = reserved = ''
    return {'SyntheticDim': dim, 'G': grid,
            'R2': regression_metrics(yt, yp)['R2'],
            'TrainTime_sec': train_time, 'InferenceTime_sec': infer_time,
            'NumParams': sum(p.numel() for p in model.parameters()),
            'PeakAllocatedMiB': allocated, 'PeakReservedMiB': reserved,
            'Device': device}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimension', type=int, choices=DIMS)
    p.add_argument('--grid', type=int, choices=GRIDS)
    p.add_argument('--device', choices=('cuda', 'cpu'), default=('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--output', type=Path, default=Path('results/generated/scalability_isolated.csv'))
    a = p.parse_args()
    if a.dimension is not None or a.grid is not None:
        if a.dimension is None or a.grid is None:
            p.error('--dimension and --grid must be given together')
        if a.device == 'cuda' and not torch.cuda.is_available():
            p.error('CUDA requested but unavailable')
        row = run_one(a.dimension, a.grid, a.device)
        print(row)
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='', encoding='utf-8') as f:
        writer = None
        for dim in DIMS:
            for grid in GRIDS:
                command = [sys.executable, '-m', 'experiments.run_scalability',
                           '--dimension', str(dim), '--grid', str(grid), '--device', a.device]
                proc = subprocess.run(command, check=True, capture_output=True, text=True)
                # Worker emits a safe Python dict with numeric or blank-string values.
                import ast
                row = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(f'd={dim}, G={grid}: R²={row["R2"]:.4f}, peak allocated MiB={row["PeakAllocatedMiB"]}', flush=True)


if __name__ == '__main__':
    main()
