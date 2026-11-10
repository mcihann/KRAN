from pathlib import Path
import pandas as pd, numpy as np
from kran.data import load_dataset
from kran.experiment_utils import build_named, run_torch_model
from kran.train import loaders, predict
from kran.metrics import classification_metrics
rows=[]
for ds in ['breast_cancer','covertype','adult_income']:
 b=load_dataset(ds,42,'minmax')
 for name in ['MLP','KAN-style','GAM','KRAN']:
  _,m=run_torch_model(name,build_named(name,b),b,42,lr=1e-3,weight_decay=1e-4)
  for sigma in [0.,.01,.05,.1]:
   rng=np.random.RandomState(42); X=np.clip(b.X_test+rng.normal(0,sigma,b.X_test.shape),0,1).astype(np.float32)
   _,te=loaders(b.X_train,b.y_train,X,b.y_test,b.batch_size,b.task); y,p,_=predict(m,te,b.task); r={'Dataset':b.name,'Model':name,'NoiseSigma':sigma,**classification_metrics(y,p)}; rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/noise_robustness.csv',index=False)
