"""Five-seed nested-holdout benchmark for KRAN and nine tabular baselines.

Run from the repository root. The outer held-out partitions are never used
for normalization, inner model selection, or training. Results and optional
KRAN checkpoints are written to the configured output directory.

Quick smoke test (not a full benchmark):
 python -m experiments.run_repeated_holdout --datasets breast_cancer --models KRAN,MLP --seeds 1042 --max-epochs 2 --output results/generated/smoke
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from kran.baselines import MLP, KANStyle, GAM, FTTransformerLite
from kran.metrics import classification_metrics, regression_metrics
from kran.model import KRAN, KRANConfig
from kran.repeated import (PRESET, TrainingPreprocessor, fingerprint, fit_full_neural,
                           load_raw_dataset, metrics_and_predictions, nested_indices,
                           output_dim, row_select, select_neural_epochs,
                           select_tree, tree_estimator)
from kran.train import set_seed

NEURAL = ('KRAN', 'MLP', 'KAN-style', 'GAM', 'FT-Transformer-lite')
TREES = ('RandomForest', 'HistGradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost')
ALL_MODELS = NEURAL + TREES


def build(name, n_features, n_outputs, task, knots=16):
    if name == 'KRAN':
        return KRAN(KRANConfig(n_features, n_outputs, task=task, spline_num_knots=knots))
    if name == 'MLP':
        return MLP(n_features, n_outputs)
    if name == 'KAN-style':
        return KANStyle(n_features, n_outputs)
    if name == 'GAM':
        return GAM(n_features, n_outputs)
    if name == 'FT-Transformer-lite':
        return FTTransformerLite(n_features, n_outputs)
    raise ValueError(name)


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(args):
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = args.datasets.split(',')
    models = args.models.split(',')
    seeds = [int(v) for v in args.seeds.split(',')]
    for name in datasets:
        if name not in PRESET:
            raise ValueError(f'Unknown dataset {name}')
    for name in models:
        if name not in ALL_MODELS:
            raise ValueError(f'Unknown model {name}')
    out = Path(args.output)
    if args.max_epochs != 150:
        print('Non-default training budget: results are not directly comparable with the standard protocol.')
    out.mkdir(parents=True, exist_ok=True)
    output_csv = out / 'run_level.csv'
    previously_run = set()
    if output_csv.exists():
        import pandas as pd
        prev = pd.read_csv(output_csv)
        previously_run = set(zip(prev['Dataset'], prev['Model'], prev['OuterSeed']))
        if not args.resume:
            raise FileExistsError(f'{output_csv} already exists. Use --resume to skip completed runs, or a new --output directory.')
    for dataset in datasets:
        raw = load_raw_dataset(dataset)
        outputs = output_dim(raw.y, raw.task)
        for seed in seeds:
            train_idx, test_idx, inner_idx, val_idx = nested_indices(raw, seed)
            inner_proc = TrainingPreprocessor().fit(row_select(raw.X, inner_idx))
            X_inner = inner_proc.transform(row_select(raw.X, inner_idx))
            X_val = inner_proc.transform(row_select(raw.X, val_idx))
            outer_proc = TrainingPreprocessor().fit(row_select(raw.X, train_idx))
            X_train = outer_proc.transform(row_select(raw.X, train_idx))
            X_test = outer_proc.transform(row_select(raw.X, test_idx))
            yi, yv, yt, yte = raw.y[inner_idx], raw.y[val_idx], raw.y[train_idx], raw.y[test_idx]
            fp_tr, fp_te = fingerprint(X_train, yt), fingerprint(X_test, yte)
            split_dir = out / 'splits'
            split_dir.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(split_dir / f'{dataset}_seed_{seed}.npz',
                                outer_train=train_idx, outer_test=test_idx,
                                inner_train=inner_idx, inner_validation=val_idx)
            for name in models:
                if (raw.name, name, seed) in previously_run:
                    print(f'Skipping completed run: {raw.name} / {name} / {seed}', flush=True)
                    continue
                set_seed(seed)
                start_select = time.perf_counter()
                npar = ''
                if name in NEURAL:
                    options = (16, 32) if name == 'KRAN' else (16,)
                    selections = []
                    for knots in options:
                        set_seed(seed)
                        model = build(name, X_inner.shape[1], outputs, raw.task, knots)
                        epoch, val_loss = select_neural_epochs(
                            model, X_inner, yi, X_val, yv, raw.task,
                            raw.batch_size, args.max_epochs, device, seed,
                        )
                        selections.append((val_loss, knots, epoch))
                    best_loss, best_knots, epoch = min(selections, key=lambda t: (t[0], t[1]))
                    selection_time = time.perf_counter() - start_select
                    set_seed(seed)
                    model = build(name, X_train.shape[1], outputs, raw.task, best_knots)
                    npar = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
                    train_seconds, target_stats = fit_full_neural(
                        model, X_train, yt, raw.task, raw.batch_size, epoch, device, seed)
                    scores, predictions, inference_seconds = metrics_and_predictions(
                        model, X_test, yte, raw.task, device, raw.batch_size, target_stats)
                    selection = f'knots={best_knots}' if name == 'KRAN' else 'epoch-selection'
                    if name == 'KRAN' and args.checkpoints:
                        state_path = out / 'checkpoints' / f'{dataset}_seed_{seed}.pt'
                        state_path.parent.mkdir(exist_ok=True, parents=True)
                        torch.save({'model_state_dict': model.cpu().state_dict(),
                                    'model_config': asdict(model.config), 'outer_seed': seed,
                                    'epoch': epoch, 'feature_names': outer_proc.feature_names,
                                    'outer_train_fingerprint': fp_tr,
                                    'outer_test_fingerprint': fp_te}, state_path)
                else:
                    if name in ('RandomForest', 'HistGradientBoosting'):
                        selection, best_loss = 'fixed', float('nan')
                        selection_time = 0.0
                    else:
                        selection, best_loss = select_tree(name, raw, X_inner, X_val, yi, yv, seed)
                        selection_time = time.perf_counter() - start_select
                    tree = tree_estimator(name, raw.task, outputs, yt, seed, selection)
                    tic = time.perf_counter()
                    y_fit = yt.ravel() if raw.task == 'regression' and outputs == 1 else yt
                    tree.fit(X_train, y_fit)
                    train_seconds = time.perf_counter() - tic
                    tic = time.perf_counter()
                    predictions = tree.predict_proba(X_test) if raw.task == 'classification' else tree.predict(X_test)
                    inference_seconds = time.perf_counter() - tic
                    scores = classification_metrics(yte, predictions) if raw.task == 'classification' else regression_metrics(yte, predictions)
                    epoch = ''
                row = {'Dataset': raw.name, 'Model': name, 'OuterSeed': seed,
                       'OuterTrainSize': len(train_idx), 'OuterTestSize': len(test_idx),
                       'EncodedFeatures': X_train.shape[1], 'Task': raw.task,
                       'Selection': selection, 'SelectionEpochs': epoch,
                       'ValidationLoss': best_loss,
                       'SelectionTime_sec': round(selection_time, 5),
                       'TrainTime_sec': round(train_seconds, 5),
                       'InferenceTime_sec': round(inference_seconds, 6),
                       'NumParams': npar, 'Device': device,
                       'OuterTrainFingerprint': fp_tr, 'OuterTestFingerprint': fp_te,
                       'MSE': scores.get('MSE', ''), 'MAE': scores.get('MAE', ''),
                       'R2': scores.get('R2', ''), 'Accuracy': scores.get('Accuracy', ''),
                       'MacroF1': scores.get('Macro-F1', ''), 'AUC': scores.get('AUC', ''),
                       'ECE': scores.get('ECE', ''), 'Brier': scores.get('Brier', ''),
                       'NLL': scores.get('NLL', '')}
                append_csv(out / 'run_level.csv', row)
                print(f'{dataset} / {name} / seed={seed}: {scores}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', default=','.join(PRESET))
    parser.add_argument('--models', default=','.join(ALL_MODELS))
    parser.add_argument('--seeds', default='1042,1043,1044,1045,1046')
    parser.add_argument('--max-epochs', default=150, type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--output', default='results/generated/repeated_holdout')
    parser.add_argument('--no-checkpoints', dest='checkpoints', action='store_false')
    parser.add_argument('--resume', action='store_true', help='Skip runs already present in output CSV')
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error('--max-epochs must be >=1')
    run(args)


if __name__ == '__main__':
    main()
