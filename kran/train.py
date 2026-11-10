import random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .regularization import decorrelation_penalty, pspline_smoothness_penalty, hsic_penalty


def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def loaders(Xtr,ytr,Xte,yte,batch_size,task):
    yt=torch.long if task=='classification' else torch.float32
    tr=TensorDataset(torch.tensor(Xtr,dtype=torch.float32),torch.tensor(ytr,dtype=yt))
    te=TensorDataset(torch.tensor(Xte,dtype=torch.float32),torch.tensor(yte,dtype=yt))
    return DataLoader(tr,batch_size=batch_size,shuffle=True), DataLoader(te,batch_size=batch_size,shuffle=False)


def train_model(model, loader, task, epochs, lr=1e-3, weight_decay=1e-4, device=None,
                lambda_decorr=0.0, lambda_hsic=0.0, gamma_smooth=0.0, hsic_max_pairs=None, hsic_max_samples=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.0) if task=='classification' else nn.MSELoss()
    t0=time.perf_counter(); history=[]
    for _ in range(epochs):
        model.train(); total=0.; n=0
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            pred,extra=model(xb,return_extras=True) if hasattr(model,'mono_map') else (model(xb),{})
            loss=loss_fn(pred,yb)
            z=extra.get('z')
            if z is not None and lambda_decorr: loss=loss+lambda_decorr*decorrelation_penalty(z)
            if z is not None and lambda_hsic: loss=loss+lambda_hsic*hsic_penalty(z[:hsic_max_samples] if hsic_max_samples else z, hsic_max_pairs)
            if gamma_smooth and hasattr(model,'head') and hasattr(model.head,'coefficients'):
                loss=loss+gamma_smooth*pspline_smoothness_penalty(model.head)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            total += loss.item()*xb.shape[0]; n += xb.shape[0]
        history.append(total/max(n,1))
    return history, time.perf_counter()-t0


def predict(model, loader, task, device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
    ys=[]; outs=[]; t0=time.perf_counter()
    with torch.no_grad():
        for xb,yb in loader:
            o=model(xb.to(device)); outs.append(o.cpu()); ys.append(yb.cpu())
    infer=time.perf_counter()-t0
    out=torch.cat(outs).numpy(); y=torch.cat(ys).numpy()
    if task=='classification':
        probs=torch.softmax(torch.tensor(out),1).numpy(); return y,probs,infer
    return y,out,infer
