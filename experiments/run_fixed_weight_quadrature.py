"""Fixed-weight KR-MonoMap quadrature convergence against G=256.

This is a post-training numerical test of checkpoint weights, NOT retraining.
Requires a checkpoint saved by run_repeated_holdout.py.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from kran.model import KRAN, KRANConfig
from kran.repeated import load_raw_dataset, nested_indices, TrainingPreprocessor, fingerprint, row_select


def calc_z(model, X, G, device):
    model.mono_map.num_quad_points = G
    model.mono_map.quad_nodes = torch.linspace(0, 1, G, device=device).view(1, -1)
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            parts.append(model.mono_map(torch.as_tensor(X[i:i+32], device=device))[0].cpu().numpy())
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('breast_cancer','covertype','adult_income',
                        'energy_efficiency','california_housing','yearpredictionmsd'), default='breast_cancer')
    p.add_argument('--seed', type=int, default=1042)
    p.add_argument('--samples', type=int, default=256, help='Maximum held-out samples (default: 256)')
    p.add_argument('--checkpoint-root', type=Path, default=Path('results/generated/repeated_holdout/checkpoints'))
    p.add_argument('--output', type=Path, default=Path('results/generated/quadrature_fixed_weights.csv'))
    a = p.parse_args()
    raw = load_raw_dataset(a.dataset)
    train_idx, test_idx, _, _ = nested_indices(raw, a.seed)
    proc = TrainingPreprocessor().fit(row_select(raw.X, train_idx))
    Xtr, Xte = proc.transform(row_select(raw.X, train_idx)), proc.transform(row_select(raw.X, test_idx))
    ckpt = torch.load(a.checkpoint_root / f'{a.dataset}_seed_{a.seed}.pt', map_location='cpu', weights_only=True)
    if fingerprint(Xtr, raw.y[train_idx]) != ckpt['outer_train_fingerprint']:
        raise RuntimeError('Checkpoint partition mismatch')
    model = KRAN(KRANConfig(**ckpt['model_config'])).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    X = Xte[:min(a.samples, len(Xte))]
    reference = calc_z(model, X, 256, 'cpu')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('Dataset','OuterSeed','G','ReferenceG',
                                                   'NumTestSamples','MaxAbsZDifference','MeanAbsZDifference'))
        if file.tell() == 0:
            writer.writeheader()
        for G in (8, 16, 32, 64):
            deviation = np.abs(calc_z(model, X, G, 'cpu') - reference)
            row = dict(Dataset=raw.name, OuterSeed=a.seed, G=G, ReferenceG=256,
                       NumTestSamples=len(X), MaxAbsZDifference=float(deviation.max()),
                       MeanAbsZDifference=float(deviation.mean()))
            writer.writerow(row)
            print(row)


if __name__ == '__main__':
    main()
