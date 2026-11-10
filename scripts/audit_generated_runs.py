"""Validate run-level integrity for repeated 6-dataset × 10-model × 5-split runs.

This checks completeness and split identity, NOT truth of reported metrics.
"""
import argparse
from pathlib import Path
import pandas as pd


MODELS = {'KRAN','MLP','KAN-style','GAM','FT-Transformer-lite','RandomForest',
          'HistGradientBoosting','XGBoost','LightGBM','CatBoost'}
DATASETS = {'Breast Cancer','Adult Income','Covertype','Energy Efficiency',
            'California Housing','YearPredictionMSD'}
SEEDS = set(range(1042, 1047))


def check(csv_file):
    df = pd.read_csv(csv_file)
    key = ['Dataset','Model','OuterSeed']
    dup = df.duplicated(key).sum()
    expected = {(d,m,s) for d in DATASETS for m in MODELS for s in SEEDS}
    observed = set(map(tuple, df[key].itertuples(index=False,name=None)))
    missing = expected - observed
    unexpected = observed - expected
    fingerprints = df.groupby(['Dataset','OuterSeed'])[['OuterTrainFingerprint',
                          'OuterTestFingerprint']].nunique(dropna=False)
    mismatches = (fingerprints > 1).sum().sum()
    print('Records:',len(df), '| Expected:',len(expected))
    print('Duplicates:',dup,'| Missing:',len(missing),'| Unexpected:',len(unexpected),
          '| Split-fingerprint mismatches:',mismatches)
    if missing:
        print('First missing entries:',sorted(missing)[:10])
    valid_metrics = df.apply(lambda r: pd.notna(r['Accuracy']) and pd.notna(r['MacroF1'])
                 if r['Task']=='classification' else
                 all(pd.notna(r[c]) for c in ('MSE','MAE','R2')),axis=1)
    print('Rows with missing required metrics:',int((~valid_metrics).sum()))
    valid = not dup and not missing and not unexpected and not mismatches and valid_metrics.all()
    if valid:
        df.groupby(['Dataset','Model'],as_index=False)[['Accuracy','MacroF1','MSE','MAE','R2']].agg(['mean','std']).to_csv(csv_file.with_name('summary_mean_std.csv'))
        print('PASS: all 300 expected entries present. Descriptive summary saved.')
    else:
        print('INCOMPLETE: expected benchmark coverage or consistency checks failed.')
    return valid


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_level_csv', type=Path)
    args = p.parse_args()
    raise SystemExit(0 if check(args.run_level_csv) else 1)
