from pathlib import Path
import pandas as pd, torch
from kran.data import load_dataset
from kran.model import KRAN, KRANConfig
from kran.experiment_utils import run_torch_model, output_dim
from kran.diagnostics import monotonicity_diagnostics, inverse_reconstruction_error
from kran.train import set_seed
rows=[]
for ds in ['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']:
 b=load_dataset(ds,42,'minmax'); d=b.X_train.shape[1]; o=output_dim(b)
 for G in [8,16,32,64]:
  name='KRAN-base' if G==32 else f'KRAN-quad-{G}'
  set_seed(42); m=KRAN(KRANConfig(d,o,task=b.task,mono_num_quad_points=G)); r,m=run_torch_model(name,m,b,42,lr=1e-3,weight_decay=1e-4)
  x=torch.tensor(b.X_test[:min(256,len(b.X_test))],dtype=torch.float32,device=next(m.parameters()).device)
  r.update(monotonicity_diagnostics(m,x)); r.update(inverse_reconstruction_error(m,x)); rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/quadrature_diagnostics.csv',index=False)
