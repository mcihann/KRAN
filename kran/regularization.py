import torch


def decorrelation_penalty(z: torch.Tensor, eps: float = 1e-6):
    if z.shape[0] <= 1:
        return z.new_tensor(0.)
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / (z.shape[0] - 1)
    std = cov.diag().clamp_min(eps).sqrt()
    corr = cov / (std[:, None] * std[None, :]).clamp_min(eps)
    eye = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    return ((corr - eye) ** 2).sum()


def pspline_smoothness_penalty(additive_head):
    w = additive_head.coefficients
    if w.shape[-1] < 3:
        return w.new_tensor(0.)
    d2 = w[..., :-2] - 2*w[..., 1:-1] + w[..., 2:]
    return (d2 ** 2).mean()


def _rbf_kernel(x, sigma=None):
    x=x.reshape(-1,1)
    dist2=(x-x.T).pow(2)
    if sigma is None:
        vals=dist2.detach()[dist2.detach()>0]
        sigma=torch.sqrt(torch.median(vals).clamp_min(1e-6)) if vals.numel() else x.new_tensor(1.)
    return torch.exp(-dist2/(2*sigma.pow(2).clamp_min(1e-12)))


def hsic_pair(x, y):
    n=x.numel()
    if n < 4: return x.new_tensor(0.)
    K=_rbf_kernel(x); L=_rbf_kernel(y)
    H=torch.eye(n,device=x.device,dtype=x.dtype)-torch.ones((n,n),device=x.device,dtype=x.dtype)/n
    return torch.trace(K@H@L@H)/((n-1)**2)


def hsic_penalty(z: torch.Tensor, max_pairs: int | None = None):
    d=z.shape[1]
    pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
    if max_pairs is not None and len(pairs)>max_pairs:
        pairs=pairs[:max_pairs]
    if not pairs: return z.new_tensor(0.)
    return torch.stack([hsic_pair(z[:,i],z[:,j]) for i,j in pairs]).mean()
