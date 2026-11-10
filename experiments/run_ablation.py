"""KRAN architecture ablations on all six datasets."""
from pathlib import Path
import pandas as pd
from kran.data import load_dataset
from kran.model import KRAN, KRANConfig
from kran.experiment_utils import run_torch_model, output_dim
from kran.train import set_seed

DATASETS=['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']
VARIANTS={
 'KRAN-base':{}, 'KRAN-integrand-exp':{'mono_integrand':'exp'}, 'KRAN-quad-8':{'mono_num_quad_points':8},
 'KRAN-quad-16':{'mono_num_quad_points':16}, 'KRAN-quad-64':{'mono_num_quad_points':64},
 'KRAN-knots-8':{'spline_num_knots':8}, 'KRAN-knots-32':{'spline_num_knots':32},
 'KRAN-dropout-0.1':{'mono_dropout':0.1}, 'KRAN-legacy-spline-MLP-head':{'head_type':'legacy_mlp'},
 'KRAN-transport-linear-head':{'head_type':'linear'}, 'IdentityTransport-additive':{'identity_transport':True},
}
rows=[]
for ds in DATASETS:
 b=load_dataset(ds,42,'minmax'); d=b.X_train.shape[1]; o=output_dim(b)
 for name,kw in VARIANTS.items():
  set_seed(42); m=KRAN(KRANConfig(d,o,task=b.task,**kw)); r,_=run_torch_model(name,m,b,42,lr=1e-3,weight_decay=1e-4); r['Variant']=r.pop('Model'); rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/ablation.csv',index=False)
