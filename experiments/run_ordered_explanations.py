"""Original-feature direct/downstream responses for 20 restart-1 checkpoints.

Match the SAME original feature across all different transport positions.
Saved checkpoints are required; model retraining is deliberately forbidden.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from kran.explanations import original_feature_responses, summarize_response
from kran.model import KRAN, KRANConfig
from kran.repeated import TrainingPreprocessor, fingerprint, load_raw_dataset, nested_indices, row_select


def run(args):
    raw=load_raw_dataset('breast_cancer')
    seeds=[int(s) for s in args.seeds.split(',')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=None
        for seed in seeds:
            tridx, _, _, _=nested_indices(raw,seed)
            proc=TrainingPreprocessor().fit(row_select(raw.X,tridx))
            Xtrain=proc.transform(row_select(raw.X,tridx))
            rng=np.random.default_rng(seed)
            refs=Xtrain[rng.choice(len(Xtrain),args.references,replace=False)]
            for ordering in ('original','reversed','random_1','random_2'):
                ckpt=torch.load(args.checkpoint_root/f'seed_{seed}_{ordering}.pt',
                                map_location='cpu', weights_only=True)
                if fingerprint(Xtrain,raw.y[tridx])!=ckpt['train_fingerprint']:
                    raise RuntimeError('Partition fingerprint mismatch')
                indices=np.asarray(ckpt['feature_order'],dtype=int)
                model=KRAN(KRANConfig(**ckpt['model_config']))
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                ordered_refs=refs[:,indices]
                for j,name in enumerate(raw.feature_names):
                    pos=int(np.where(indices==j)[0][0])
                    lo,hi=np.quantile(Xtrain[:,j],[.05,.95])
                    grid=np.linspace(lo,hi,args.points,dtype=np.float32)
                    result=original_feature_responses(model,ordered_refs,pos,grid)
                    if result['decomposition_max_error']>1e-4:
                        raise ValueError('Original-feature decomposition failed')
                    summary=summarize_response(result,grid)
                    for k in range(args.points):
                        row={'OuterSeed':seed,'Restart':1,'Ordering':ordering,
                             'OriginalFeatureIndex':j,'OriginalFeatureName':name,
                             'PositionInTransport':pos,'ScaledValue':float(grid[k]),
                             'DirectMean':float(summary['direct_mean'][k]),
                             'DownstreamMean':float(summary['downstream_mean'][k]),
                             'TotalMean':float(summary['total_mean'][k]),
                             'MaxDecompositionResidual':result['decomposition_max_error']}
                        if writer is None:
                            writer=csv.DictWriter(f,fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                print(seed,ordering,'evaluated',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root',type=Path,default=Path('results/generated/controlled_ordering/restart_1_checkpoints'))
    p.add_argument('--output',type=Path,default=Path('results/generated/ordered_explanations.csv'))
    p.add_argument('--seeds',default='1042,1043,1044,1045,1046')
    p.add_argument('--references',type=int,default=48)
    p.add_argument('--points',type=int,default=31)
    run(p.parse_args())
