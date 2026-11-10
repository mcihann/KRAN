from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr, kendalltau
from kran.data import load_dataset
from kran.model import KRAN, KRANConfig
from kran.experiment_utils import run_torch_model, output_dim
from kran.diagnostics import additive_feature_importance, latent_effect_curves
from kran.train import set_seed
rows=[]
for ds in ['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']:
 b=load_dataset(ds,42,'minmax'); imps=[]; curves=[]
 for seed in [42,43,44]:
  set_seed(seed); m=KRAN(KRANConfig(b.X_train.shape[1],output_dim(b),task=b.task)); _,m=run_torch_model('KRAN',m,b,seed,lr=1e-3,weight_decay=1e-4); imps.append(additive_feature_importance(m)); curves.append(latent_effect_curves(m,101)[1])
 sp=[]; kt=[]; ov=[]
 for i in range(3):
  for j in range(i+1,3):
   sp.append(spearmanr(imps[i],imps[j]).statistic); kt.append(kendalltau(imps[i],imps[j]).statistic)
   A=set(np.argsort(-imps[i])[:min(10,len(imps[i]))]); B=set(np.argsort(-imps[j])[:min(10,len(imps[j]))]); ov.append(len(A&B)/len(A))
 rows.append({'Dataset':b.name,'Spearman':np.mean(sp),'KendallTau':np.mean(kt),'Top10Overlap':np.mean(ov)}); print(rows[-1])
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/interpretability_stability.csv',index=False)
