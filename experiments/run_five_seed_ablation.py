"""Five-seed, three-variant fixed-budget architectural comparison.

Breast Cancer: 5 split seeds 42--46, 50 epochs, batch 32.
California Housing: 5 split seeds 42--46, 30 epochs, batch 256.
Do not combine these measurements with validation-selected 1042--1046 runs.
"""
import argparse
from pathlib import Path
import pandas as pd
from kran.data import load_dataset
from kran.experiment_utils import run_torch_model, output_dim
from kran.model import KRAN, KRANConfig
from kran.train import set_seed

DATASETS=('breast_cancer','california_housing')
VARIANTS={
    'IdentityTransport-additive': {'identity_transport':True},
    'KRAN-legacy-spline-MLP-head': {'head_type':'legacy_mlp'},
    'KRAN-base': {},
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',default='42,43,44,45,46')
    p.add_argument('--datasets',default=','.join(DATASETS))
    p.add_argument('--output',type=Path,default=Path('results/generated/five_seed_ablation.csv'))
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'{a.output} exists; use a new --output to keep runs distinct')
    results=[]
    for ds in a.datasets.split(','):
        if ds not in DATASETS: raise ValueError(ds)
        for seed in [int(x) for x in a.seeds.split(',')]:
            bundle=load_dataset(ds,seed=seed,scaler='minmax')
            for variant,options in VARIANTS.items():
                set_seed(seed)
                model=KRAN(KRANConfig(bundle.X_train.shape[1],output_dim(bundle),
                                      task=bundle.task,**options))
                row,_=run_torch_model(variant,model,bundle,seed=seed,
                                     lr=1e-3,weight_decay=1e-4)
                row['Variant']=row.pop('Model')
                row['OuterSeed']=seed
                row['Protocol']='fixed_epoch_architecture_42_46'
                row['Epochs']=bundle.epochs
                results.append(row)
                print(ds,seed,variant,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(a.output,index=False)
    print('Saved',len(results),'rows to',a.output)


if __name__=='__main__':
    main()
