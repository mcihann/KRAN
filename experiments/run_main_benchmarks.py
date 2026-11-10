"""Main six-dataset benchmark. Downloads public datasets on first use."""
from pathlib import Path
import time, pandas as pd
from kran.data import load_dataset
from kran.experiment_utils import build_named, run_torch_model
from kran.baselines import sklearn_tree
from kran.metrics import classification_metrics, regression_metrics

DATASETS=['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']
MODELS=['MLP','KAN-style','GAM','FT-Transformer-lite','KRAN']
rows=[]
for ds in DATASETS:
    b=load_dataset(ds,seed=42,scaler='minmax')
    for name in MODELS:
        r,_=run_torch_model(name,build_named(name,b),b,seed=42,lr=1e-3,weight_decay=1e-4); rows.append(r); print(r)
    for name in ['HistGradientBoosting','RandomForest']:
        m=sklearn_tree(name,b.task,b.y_train,42); t=time.perf_counter(); m.fit(b.X_train,b.y_train); tt=time.perf_counter()-t
        t=time.perf_counter(); pred=m.predict(b.X_test); it=time.perf_counter()-t
        if b.task=='classification':
            probs=m.predict_proba(b.X_test); met=classification_metrics(b.y_test,probs)
        else: met=regression_metrics(b.y_test,pred)
        r={'Dataset':b.name,'Model':name,'#Params':None,'TrainTime_sec':tt,'InferenceTime_sec':it,**met}; rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/main_benchmarks.csv',index=False)
