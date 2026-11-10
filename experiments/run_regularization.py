from pathlib import Path
import pandas as pd, torch
from kran.data import load_dataset
from kran.model import KRAN, KRANConfig
from kran.experiment_utils import run_torch_model, output_dim
from kran.train import set_seed
from kran.regularization import hsic_penalty
rows=[]
settings={
 'KRAN-no-penalty':{},
 'KRAN-corr-decorrelation':{'lambda_decorr':1e-3},
 'KRAN-hsic-dependence':{'lambda_hsic':1e-3,'hsic_max_pairs':64,'hsic_max_samples':64},
 'KRAN-corr+smoothness':{'lambda_decorr':1e-3,'gamma_smooth':1e-3},
}
for ds in ['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']:
 b=load_dataset(ds,42,'minmax'); d=b.X_train.shape[1]; o=output_dim(b)
 for name,kw in settings.items():
  set_seed(42); m=KRAN(KRANConfig(d,o,task=b.task)); r,m=run_torch_model(name,m,b,42,lr=1e-3,weight_decay=1e-4,**kw)
  with torch.no_grad():
   z,_=m.mono_map(torch.tensor(b.X_test[:min(1024,len(b.X_test))],dtype=torch.float32,device=next(m.parameters()).device))
   c=torch.corrcoef(z.T).abs(); mask=~torch.eye(c.shape[0],dtype=torch.bool,device=c.device); r['FinalLatentAbsCorr']=float(torch.nan_to_num(c[mask],nan=0).mean()) if mask.any() else 0.
   r['SampledHSIC']=float(hsic_penalty(z[:min(128,len(z))],max_pairs=64))
  rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/regularization.csv',index=False)
