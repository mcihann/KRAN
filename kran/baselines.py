from typing import Sequence
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from .model import SplineExpansion, AdditiveSplineHead


class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden=(64,64)):
        super().__init__(); layers=[]; d=input_dim
        for h in hidden: layers += [nn.Linear(d,h),nn.SiLU()]; d=h
        layers.append(nn.Linear(d,output_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


class KANStyle(nn.Module):
    """Matched spline-based neural baseline; not a full KAN implementation."""
    def __init__(self,input_dim,output_dim,num_knots=16,hidden=(64,64)):
        super().__init__(); self.spline=SplineExpansion(input_dim,num_knots); layers=[]; d=input_dim*num_knots
        for h in hidden: layers += [nn.Linear(d,h),nn.SiLU()]; d=h
        layers.append(nn.Linear(d,output_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(self.spline(x).flatten(1))


class GAM(nn.Module):
    """Raw-input strictly additive spline baseline matched to the KRAN head."""
    def __init__(self,input_dim,output_dim,num_knots=16):
        super().__init__(); self.head=AdditiveSplineHead(input_dim,output_dim,num_knots)
    def forward(self,x): return self.head(x)


class FTTransformerLite(nn.Module):
    def __init__(self,input_dim,output_dim,d_token=32,nhead=4,num_layers=2):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(input_dim,d_token)*0.02)
        self.bias=nn.Parameter(torch.zeros(input_dim,d_token))
        enc=nn.TransformerEncoderLayer(d_model=d_token,nhead=nhead,dim_feedforward=64,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(enc,num_layers=num_layers); self.out=nn.Linear(d_token,output_dim)
    def forward(self,x):
        tok=x.unsqueeze(-1)*self.weight.unsqueeze(0)+self.bias.unsqueeze(0)
        return self.out(self.encoder(tok).mean(1))


def sklearn_tree(name, task, y_train=None, seed=42):
    if task=='classification':
        return RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=seed) if name=='RandomForest' else HistGradientBoostingClassifier(random_state=seed)
    base=RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=seed) if name=='RandomForest' else HistGradientBoostingRegressor(random_state=seed)
    if y_train is not None and getattr(y_train,'ndim',1)>1 and y_train.shape[1]>1 and name=='HistGradientBoosting':
        return MultiOutputRegressor(base)
    return base
