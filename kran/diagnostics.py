import numpy as np
import torch


def monotonicity_diagnostics(model, x, eps=1e-4, tol=1e-8):
    model.eval(); x=x.detach().clone(); rates=[]; derivs=[]
    with torch.no_grad():
        z0,_=model.mono_map(x)
        for j in range(x.shape[1]):
            xp=x.clone(); xp[:,j]=(xp[:,j]+eps).clamp(max=1.)
            z1,_=model.mono_map(xp)
            dz=(z1[:,j]-z0[:,j])/(xp[:,j]-x[:,j]).clamp_min(1e-12)
            rates.append(float((dz < -tol).float().mean())); derivs.append(dz.cpu().numpy())
    d=np.concatenate(derivs)
    return {'MonotonicityViolation':float(np.mean(rates)),'MinFiniteDiffDerivative':float(d.min()),'MeanFiniteDiffDerivative':float(d.mean())}


def inverse_monotone_map(model, z_target, n_iter=40):
    """Sequential bisection inversion of the triangular KR-MonoMap."""
    mm=model.mono_map; B,d=z_target.shape; x=torch.zeros_like(z_target)
    with torch.no_grad():
        for j in range(d):
            lo=torch.zeros(B,device=z_target.device); hi=torch.ones(B,device=z_target.device)
            for _ in range(n_iter):
                mid=(lo+hi)/2; trial=x.clone(); trial[:,j]=mid
                # Later coordinates do not affect z_j; keep them zero.
                z,_=mm(trial); v=z[:,j]
                lo=torch.where(v<z_target[:,j],mid,lo); hi=torch.where(v>=z_target[:,j],mid,hi)
            x[:,j]=(lo+hi)/2
    return x


def inverse_reconstruction_error(model, x, n_iter=40):
    with torch.no_grad(): z,_=model.mono_map(x); xr=inverse_monotone_map(model,z,n_iter)
    e=(xr-x).abs(); return {'InverseReconMAE':float(e.mean()),'InverseReconMaxAE':float(e.max())}


def additive_feature_importance(model):
    w=model.head.coefficients.detach().abs().mean(dim=(0,2))
    return (w/(w.sum()+1e-12)).cpu().numpy()


def latent_effect_curves(model, grid_points=101):
    if not hasattr(model.head,'coefficients'): raise TypeError('Additive head required')
    g=torch.linspace(0,1,grid_points,device=next(model.parameters()).device)
    curves=[]
    with torch.no_grad():
        for j in range(model.config.input_dim):
            phi=model.head.spline.bases[j](g)
            # [grid,O]
            eff=torch.einsum('gm,om->go',phi,model.head.coefficients[:,j,:])
            curves.append(eff.cpu().numpy())
    return g.cpu().numpy(), curves
