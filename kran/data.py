from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.datasets import load_breast_cancer, fetch_covtype, fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


@dataclass
class DatasetBundle:
    name: str; task: str; X_train: np.ndarray; X_test: np.ndarray; y_train: np.ndarray; y_test: np.ndarray
    feature_names: list[str]; num_classes: int | None; batch_size: int; epochs: int


class RobustMinMax:
    def fit(self,X):
        self.lo=np.nanpercentile(X,1,axis=0); self.hi=np.nanpercentile(X,99,axis=0); return self
    def transform(self,X):
        X=np.clip(X,self.lo,self.hi); return np.clip((X-self.lo)/(self.hi-self.lo+1e-12),0,1)
    def fit_transform(self,X): return self.fit(X).transform(X)


class StandardCDF:
    def fit(self,X): self.scaler=StandardScaler().fit(X); return self
    def transform(self,X): return ndtr(self.scaler.transform(X)).astype(np.float32)
    def fit_transform(self,X): return self.fit(X).transform(X)


def _scaler(kind, n_train):
    if kind=='minmax': return MinMaxScaler(clip=True)
    if kind=='robust_minmax': return RobustMinMax()
    if kind=='quantile_uniform': return QuantileTransformer(n_quantiles=min(1000,n_train),output_distribution='uniform',random_state=42)
    if kind=='standard_cdf': return StandardCDF()
    raise ValueError(kind)


def _scale_after_split(Xtr,Xte,kind):
    s=_scaler(kind,len(Xtr)); return s.fit_transform(Xtr).astype(np.float32), s.transform(Xte).astype(np.float32), s


def load_dataset(name: str, seed=42, scaler='minmax'):
    key=name.lower().replace(' ','_')
    if key in {'breast_cancer','breast'}:
        d=load_breast_cancer(); X=d.data.astype(np.float32); y=d.target.astype(np.int64); names=list(d.feature_names)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y)
        Xtr,Xte,_=_scale_after_split(Xtr,Xte,scaler); return DatasetBundle('Breast Cancer','classification',Xtr,Xte,ytr,yte,names,2,32,50)
    if key in {'covertype','covtype'}:
        d=fetch_covtype(); X=d.data.astype(np.float32); y=d.target.astype(np.int64)-1
        if len(X)>50000:
            idx=np.random.RandomState(42).choice(len(X),50000,replace=False); X=X[idx]; y=y[idx]
        names=[f'feature_{i}' for i in range(X.shape[1])]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y)
        Xtr,Xte,_=_scale_after_split(Xtr,Xte,scaler); return DatasetBundle('Covertype','classification',Xtr,Xte,ytr,yte,names,7,256,10)
    if key in {'adult','adult_income'}:
        d=fetch_openml('adult',version=2,as_frame=True); X=d.data.copy(); y=(d.target.astype(str).str.contains('>50K')).astype(int).to_numpy()
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y)
        cats=list(X.select_dtypes(exclude=np.number).columns); nums=[c for c in X.columns if c not in cats]
        ct=ColumnTransformer([('num','passthrough',nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
        A=ct.fit_transform(Xtr).astype(np.float32); B=ct.transform(Xte).astype(np.float32)
        names=list(ct.get_feature_names_out()); A,B,_=_scale_after_split(A,B,scaler)
        return DatasetBundle('Adult Income','classification',A,B,ytr,yte,names,2,256,10)
    if key in {'energy','energy_efficiency'}:
        d=fetch_openml('energy_efficiency',as_frame=True); X=d.data.to_numpy(np.float32); y=d.target.to_numpy(np.float32)
        if y.ndim==1:y=y[:,None]
        names=list(d.data.columns); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed)
        Xtr,Xte,_=_scale_after_split(Xtr,Xte,scaler); return DatasetBundle('Energy Efficiency','regression',Xtr,Xte,ytr,yte,names,None,32,100)
    if key in {'california','california_housing'}:
        d=fetch_california_housing(); X=d.data.astype(np.float32); y=d.target.astype(np.float32)[:,None]
        names=list(d.feature_names); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed)
        Xtr,Xte,_=_scale_after_split(Xtr,Xte,scaler); return DatasetBundle('California Housing','regression',Xtr,Xte,ytr,yte,names,None,256,30)
    if key in {'year','yearpredictionmsd'}:
        d=fetch_openml(data_id=44027,as_frame=False); X=d.data.astype(np.float32); y=d.target.astype(np.float32)[:,None]
        X=X[:50000]; y=y[:50000]; names=[f'feature_{i}' for i in range(X.shape[1])]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed)
        Xtr,Xte,_=_scale_after_split(Xtr,Xte,scaler); return DatasetBundle('YearPredictionMSD','regression',Xtr,Xte,ytr,yte,names,None,256,20)
    raise ValueError(f'Unknown dataset: {name}')
