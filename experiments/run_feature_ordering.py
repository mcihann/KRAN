from pathlib import Path
import pandas as pd, numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from kran.data import load_dataset
from kran.model import KRAN, KRANConfig
from kran.experiment_utils import run_torch_model, output_dim
from kran.train import set_seed
rows=[]
for ds in ['breast_cancer','covertype','adult_income','energy_efficiency','california_housing','yearpredictionmsd']:
 b=load_dataset(ds,42,'minmax'); X=b.X_train; y=b.y_train[:,0] if b.task=='regression' and b.y_train.ndim>1 else b.y_train.ravel()
 mi=(mutual_info_classif(X,y,random_state=42) if b.task=='classification' else mutual_info_regression(X,y,random_state=42));
 rf=(RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1) if b.task=='classification' else RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)).fit(X,y)
 corr=np.array([abs(np.corrcoef(X[:,j],y)[0,1]) if np.std(X[:,j])>0 else 0 for j in range(X.shape[1])]); rng=np.random.RandomState(42)
 orders={'original':np.arange(X.shape[1]),'mutual_info':np.argsort(-mi),'rf_importance':np.argsort(-rf.feature_importances_),'target_corr':np.argsort(-np.nan_to_num(corr))}
 for k in range(1,4): orders[f'random_{k}']=rng.permutation(X.shape[1])
 for name,idx in orders.items():
  b2=type(b)(b.name,b.task,b.X_train[:,idx],b.X_test[:,idx],b.y_train,b.y_test,[b.feature_names[i] for i in idx],b.num_classes,b.batch_size,b.epochs)
  set_seed(42); m=KRAN(KRANConfig(b2.X_train.shape[1],output_dim(b2),task=b2.task)); r,_=run_torch_model('KRAN',m,b2,42,lr=1e-3,weight_decay=1e-4); r['Ordering']=name; rows.append(r); print(r)
Path('results/generated').mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv('results/generated/feature_ordering.csv',index=False)
