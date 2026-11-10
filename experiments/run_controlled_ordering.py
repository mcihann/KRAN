"""40-run Breast Cancer feature-ordering sensitivity (new rerunnable protocol).

5 outer seeds × 2 initialization restarts × 4 orders. Each condition reuses
original-order-selected knots/epochs, NOT separately retuned per order.
Requires results/generated/repeated_holdout/run_level.csv for KRAN.
Random permutations are deterministically generated from the experiment seeds.
The same feature identities are tracked across all input orderings.
"""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kran.metrics import classification_metrics
from kran.model import KRAN, KRANConfig
from kran.repeated import TrainingPreprocessor, fingerprint, load_raw_dataset, nested_indices, row_select
from kran.train import loaders, predict, set_seed, train_model


def run(args):
    runs = pd.read_csv(args.original_runs)
    raw = load_raw_dataset('breast_cancer')
    seeds = [int(s) for s in args.seeds.split(',')]
    restarts = [int(s) for s in args.restarts.split(',')]
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / 'predictive_ordering.csv'
    if result_file.exists():
        raise FileExistsError('Use a new --output directory to avoid mixing runs')
    with result_file.open('w', newline='', encoding='utf-8') as handle:
        writer = None
        for seed in seeds:
            prior = runs[(runs['Dataset'] == 'Breast Cancer') &
                         (runs['Model'] == 'KRAN') & (runs['OuterSeed'] == seed)]
            if len(prior) != 1:
                raise ValueError(f'Missing or ambiguous selected original-order run: {seed}')
            knots = int(prior.iloc[0]['Selection'].split('=')[1])
            epochs = int(prior.iloc[0]['SelectionEpochs'])
            tridx, teidx, _, _ = nested_indices(raw, seed)
            proc = TrainingPreprocessor().fit(row_select(raw.X, tridx))
            Xtr = proc.transform(row_select(raw.X, tridx))
            Xte = proc.transform(row_select(raw.X, teidx))
            d = Xtr.shape[1]
            order_rng = np.random.RandomState(1000 + seed)
            orders = {'original': np.arange(d), 'reversed': np.arange(d)[::-1],
                      'random_1': order_rng.permutation(d), 'random_2': order_rng.permutation(d)}
            for restart in restarts:
                init_seed = seed * 10 + restart
                for order_name, idx in orders.items():
                    set_seed(init_seed)
                    cfg = KRANConfig(d, 2, task='classification', spline_num_knots=knots)
                    model = KRAN(cfg)
                    train_loader, test_loader = loaders(Xtr[:,idx], raw.y[tridx],
                                                        Xte[:,idx], raw.y[teidx], 32, 'classification')
                    _, train_time = train_model(model, train_loader, 'classification',
                                                epochs, lr=1e-3, weight_decay=1e-4,
                                                device=args.device)
                    y, probs, inference_time = predict(model, test_loader, 'classification', device=args.device)
                    metrics = classification_metrics(y, probs)
                    row = {'OuterSeed':seed, 'Restart':restart, 'Ordering':order_name,
                           'SelectedKnots':knots, 'SelectedEpochs':epochs,
                           'TrainTime_sec':train_time, 'InferenceTime_sec':inference_time,
                           'TrainFingerprint':fingerprint(Xtr, raw.y[tridx]),
                           'TestFingerprint':fingerprint(Xte, raw.y[teidx]),
                           'Accuracy':metrics['Accuracy'],'MacroF1':metrics['Macro-F1'],
                           'AUC':metrics['AUC'], 'NLL':metrics['NLL']}
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    if restart == 1:
                        save_to = args.output/'restart_1_checkpoints'/f'seed_{seed}_{order_name}.pt'
                        save_to.parent.mkdir(parents=True,exist_ok=True)
                        torch.save({'model_state_dict':model.cpu().state_dict(),
                            'model_config':asdict(cfg), 'outer_seed':seed,
                            'restart':restart,'feature_order':idx.tolist(),
                            'original_feature_names':[str(n) for n in raw.feature_names],
                            'train_fingerprint':fingerprint(Xtr,raw.y[tridx])},save_to)
                    print(seed,restart,order_name,metrics['Macro-F1'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-runs',type=Path,default=Path('results/generated/repeated_holdout/run_level.csv'))
    p.add_argument('--output',type=Path,default=Path('results/generated/controlled_ordering'))
    p.add_argument('--seeds',default='1042,1043,1044,1045,1046')
    p.add_argument('--restarts',default='0,1')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    run(p.parse_args())
