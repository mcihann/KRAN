from pathlib import Path
import time, numpy as np, pandas as pd, torch
from .model import KRAN, KRANConfig
from .baselines import MLP, KANStyle, GAM, FTTransformerLite, sklearn_tree
from .train import loaders, train_model, predict, set_seed
from .metrics import classification_metrics, regression_metrics


def output_dim(bundle): return bundle.num_classes if bundle.task=='classification' else (bundle.y_train.shape[1] if bundle.y_train.ndim>1 else 1)


def run_torch_model(name, model, bundle, seed=42, device=None, **train_kw):
    set_seed(seed); tr,te=loaders(bundle.X_train,bundle.y_train,bundle.X_test,bundle.y_test,bundle.batch_size,bundle.task)
    hist,tt=train_model(model,tr,bundle.task,bundle.epochs,device=device,**train_kw)
    y,p,it=predict(model,te,bundle.task,device=device)
    met=classification_metrics(y,p) if bundle.task=='classification' else regression_metrics(y,p)
    return {'Dataset':bundle.name,'Model':name,'#Params':sum(q.numel() for q in model.parameters() if q.requires_grad),'TrainTime_sec':tt,'InferenceTime_sec':it,**met},model


def build_named(name,bundle,seed=42,**kran_overrides):
    set_seed(seed)
    d=bundle.X_train.shape[1]; o=output_dim(bundle)
    if name=='KRAN': return KRAN(KRANConfig(d,o,task=bundle.task,**kran_overrides))
    if name=='MLP': return MLP(d,o)
    if name=='KAN-style': return KANStyle(d,o)
    if name=='GAM': return GAM(d,o)
    if name=='FT-Transformer-lite': return FTTransformerLite(d,o)
    raise ValueError(name)
