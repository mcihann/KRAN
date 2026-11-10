"""IG, permutation-NLL and optional approximate SHAP for saved Breast KRAN.

IG uses a training-derived mean baseline and reports numerical completeness.
SHAP, when enabled, uses a fixed 16-profile background with Kernel SHAP;
its sampling budget is configurable, and explanations remain model-dependent.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from kran.attributions import integrated_gradients, permutation_nll
from kran.model import KRAN, KRANConfig
from kran.repeated import TrainingPreprocessor, fingerprint, load_raw_dataset, nested_indices, row_select


def logits_margin_function(model, input_features):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_features), 64):
            y = model(torch.tensor(input_features[i:i+64], dtype=torch.float32))
            out.extend((y[:, 0] - y[:, 1]).numpy())
    return np.asarray(out)


def run(args):
    raw = load_raw_dataset('breast_cancer')
    train_idx, test_idx, _, _ = nested_indices(raw, args.seed)
    preprocess = TrainingPreprocessor().fit(row_select(raw.X, train_idx))
    Xtr = preprocess.transform(row_select(raw.X, train_idx))
    Xte = preprocess.transform(row_select(raw.X, test_idx))
    ckpt = torch.load(args.checkpoint_root / f'breast_cancer_seed_{args.seed}.pt',
                      map_location='cpu', weights_only=True)
    if fingerprint(Xtr, raw.y[train_idx]) != ckpt['outer_train_fingerprint']:
        raise RuntimeError('Dataset mismatch with saved model')
    model = KRAN(KRANConfig(**ckpt['model_config']))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    rng = np.random.default_rng(args.seed)
    explained_idx = rng.choice(len(Xte), size=min(args.samples, len(Xte)), replace=False)
    explained = Xte[explained_idx]
    baseline = Xtr.mean(axis=0)
    IG, residual = integrated_gradients(model, explained, baseline, steps=args.steps,
                                        output='class0_margin', chunk_size=args.ig_chunk_size)
    perm_X, perm_y = Xte, raw.y[test_idx]
    if args.permutation_test_samples is not None:
        chosen = rng.choice(len(Xte), min(args.permutation_test_samples, len(Xte)), replace=False)
        perm_X, perm_y = Xte[chosen], raw.y[test_idx][chosen]
    perm_mean, perm_std = permutation_nll(model, perm_X, perm_y,
                                          repetitions=args.permutation_repetitions, seed=args.seed)
    shap_scores = np.full(Xte.shape[1], np.nan)
    shap_residual = float('nan')
    if args.shap:
        import shap
        bidx = rng.choice(len(Xtr), size=min(16, len(Xtr)), replace=False)
        background = Xtr[bidx]
        explainer = shap.KernelExplainer(lambda v: logits_margin_function(model, np.asarray(v, dtype=np.float32)),
                                          background)
        values = np.asarray(explainer.shap_values(explained, nsamples=args.shap_nsamples))
        if values.shape != explained.shape:
            raise ValueError(f'Unexpected SHAP values shape {values.shape}; please inspect SHAP version')
        shap_scores = np.abs(values).mean(axis=0)
        predicted = logits_margin_function(model, explained)
        shap_residual = float(np.max(np.abs(predicted - (values.sum(axis=1) + float(explainer.expected_value)))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['OuterSeed', 'FeatureIndex', 'Feature', 'IGMeanAbsolute',
                   'PermutationNLLMean', 'PermutationNLLStd', 'SHAPMeanAbsolute',
                   'IGMaxAbsCompletenessResidual', 'SHAPMaxAbsAdditivityResidual'])
        writer.writeheader()
        for j, name in enumerate(raw.feature_names):
            writer.writerow({'OuterSeed': args.seed, 'FeatureIndex': j, 'Feature': name,
                 'IGMeanAbsolute': float(np.abs(IG[:, j]).mean()),
                 'PermutationNLLMean': float(perm_mean[j]),
                 'PermutationNLLStd': float(perm_std[j]),
                 'SHAPMeanAbsolute': float(shap_scores[j]),
                 'IGMaxAbsCompletenessResidual': float(np.abs(residual).max()),
                 'SHAPMaxAbsAdditivityResidual': shap_residual})
    print(f'Wrote {args.output}; IG max absolute completeness residual: {np.abs(residual).max():.6g}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=1042)
    p.add_argument('--checkpoint-root', type=Path, default=Path('results/generated/repeated_holdout/checkpoints'))
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--ig-chunk-size', type=int, default=8)
    p.add_argument('--permutation-repetitions', type=int, default=10)
    p.add_argument('--permutation-test-samples', type=int, default=None, help='Debug only: subsample permutation test set')
    p.add_argument('--shap', action='store_true', help='Enable approximate Kernel SHAP (computationally expensive)')
    p.add_argument('--shap-nsamples', type=int, default=128)
    p.add_argument('--output', type=Path, default=Path('results/generated/breast_original_feature_attributions.csv'))
    run(p.parse_args())
