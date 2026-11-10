from pathlib import Path
import pandas as pd
from kran.data import load_dataset
from kran.model import KRAN, KRANConfig
from kran.experiment_utils import run_torch_model, output_dim
from kran.train import set_seed
rows=[]
for ds in ['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']:
 for s in ['minmax','robust_minmax','quantile_uniform','standard_cdf']:
  b=load_dataset(ds,42,s); set_seed(42); m=KRAN(KRANConfig(b.X_train.shape[1],output_dim(b),task=b.task)); r,_=run_torch_model('KRAN',m,b,42,lr=1e-3,weight_decay=1e-4); r['Scalar']=s; rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/normalization.csv',index=False)
