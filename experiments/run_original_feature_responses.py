"""Reference-conditioned Breast Cancer original-feature response decomposition.

Requires checkpoint(s) from ``python -m experiments.run_repeated_holdout``.
Output curves are model-response diagnostics, NOT causal effects or confidence intervals.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from kran.explanations import original_feature_responses, summarize_response
from kran.model import KRAN, KRANConfig
from kran.repeated import (TrainingPreprocessor, fingerprint, load_raw_dataset,
                           nested_indices, row_select)


def run(out, checkpoint_root, seeds, num_references, num_points):
    raw = load_raw_dataset('breast_cancer')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = None
        for seed in seeds:
            train_idx, test_idx, _, _ = nested_indices(raw, seed)
            proc = TrainingPreprocessor().fit(row_select(raw.X, train_idx))
            X_train = proc.transform(row_select(raw.X, train_idx))
            X_test = proc.transform(row_select(raw.X, test_idx))
            path = checkpoint_root / f'breast_cancer_seed_{seed}.pt'
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            model = KRAN(KRANConfig(**ckpt['model_config']))
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            actual = fingerprint(X_train, raw.y[train_idx])
            if actual != ckpt['outer_train_fingerprint']:
                raise RuntimeError('Training-partition fingerprint mismatch: cannot use this checkpoint')
            rng = np.random.default_rng(seed)
            references = X_train[rng.choice(len(X_train), size=num_references, replace=False)]
            for j, name in enumerate(raw.feature_names):
                low, high = np.quantile(X_train[:, j], [.05, .95])
                grid = np.linspace(low, high, num_points, dtype=np.float32)
                response = original_feature_responses(model, references, j, grid)
                summary = summarize_response(response, grid)
                dummy = np.zeros((num_points, X_train.shape[1]), dtype=np.float32)
                dummy[:, j] = grid
                original_units = proc.scaler.inverse_transform(dummy)[:, j]
                if response['decomposition_max_error'] > 1e-4:
                    raise RuntimeError('Direct + downstream failed to reconstruct total response')
                for i in range(num_points):
                    row = {'OuterSeed': seed, 'FeatureIndex': j, 'Feature': name,
                           'ScaledValue': float(grid[i]), 'OriginalUnitsValue': float(original_units[i]),
                           'NumReferenceProfiles': num_references,
                           'MaxDecompositionResidual': response['decomposition_max_error']}
                    row.update({k: float(v[i]) for k, v in summary.items() if k != 'scaled_value'})
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
            print(f'Breast Cancer original-input responses completed for outer seed {seed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', default='1042,1043,1044,1045,1046')
    p.add_argument('--checkpoint-root', type=Path, default=Path('results/generated/repeated_holdout/checkpoints'))
    p.add_argument('--output', type=Path, default=Path('results/generated/breast_original_feature_responses.csv'))
    p.add_argument('--references', type=int, default=48)
    p.add_argument('--points', type=int, default=31)
    a = p.parse_args()
    run(a.output, a.checkpoint_root, [int(v) for v in a.seeds.split(',')], a.references, a.points)
