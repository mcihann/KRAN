from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _activation(name: str):
    name = name.lower()
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "swish": nn.SiLU, "tanh": nn.Tanh, "elu": nn.ELU}[name]


class ConditionalScalarNet(nn.Module):
    """s_j(x_{1:j-1}, t), used inside the monotone triangular transport."""
    def __init__(self, in_dim: int, hidden_dims: Sequence[int] = (64, 64),
                 activation=nn.SiLU, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KRMonoMap(nn.Module):
    r"""Normalized Knothe-Rosenblatt-inspired monotone triangular map.

    For coordinate j:

        z_j = int_0^{x_j} rho(s_j(x_{<j}, t)) dt /
              int_0^1     rho(s_j(x_{<j}, t)) dt

    The positive integrand makes z_j monotone in x_j. Both numerator and denominator are evaluated by trapezoidal quadrature.
    """
    def __init__(self, dim: int, hidden_dims: Sequence[int] = (64, 64),
                 activation=nn.SiLU, integrand: str = "softplus",
                 num_quad_points: int = 32, dropout: float = 0.0, eps: float = 1e-12):
        super().__init__()
        if num_quad_points < 2:
            raise ValueError("num_quad_points must be >= 2")
        if integrand not in {"softplus", "exp"}:
            raise ValueError("integrand must be 'softplus' or 'exp'")
        self.dim = dim
        self.integrand = integrand
        self.num_quad_points = num_quad_points
        self.eps = eps
        self.s_nets = nn.ModuleList([
            ConditionalScalarNet(j + 1, hidden_dims, activation, dropout)
            for j in range(dim)
        ])
        self.register_buffer("quad_nodes", torch.linspace(0., 1., num_quad_points).view(1, -1))

    def _positive(self, s):
        return torch.exp(s) if self.integrand == "exp" else F.softplus(s)

    def _score_on_grid(self, j, context, grid):
        B, K = grid.shape
        if context is None:
            features = grid.unsqueeze(-1)
        else:
            ctx = context.unsqueeze(1).expand(-1, K, -1)
            features = torch.cat([ctx, grid.unsqueeze(-1)], dim=-1)
        scores = self.s_nets[j](features.reshape(B * K, -1)).reshape(B, K)
        return self._positive(scores)

    @staticmethod
    def _trapz_uniform(vals, upper):
        # vals: [B,K] sampled uniformly on [0, upper]
        K = vals.shape[1]
        dx = upper.squeeze(-1) / float(K - 1)
        return (0.5 * (vals[:, 0] + vals[:, -1]) + vals[:, 1:-1].sum(1)) * dx

    def forward(self, x, return_components: bool = False):
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected [B,{self.dim}], got {tuple(x.shape)}")
        x = x.clamp(0., 1.)
        B = x.shape[0]
        z = torch.empty_like(x)
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)
        denoms = []
        for j in range(self.dim):
            context = None if j == 0 else x[:, :j]
            xj = x[:, j:j+1]
            unit = self.quad_nodes.to(dtype=x.dtype).expand(B, -1)

            # denominator on [0,1]
            g_den = self._score_on_grid(j, context, unit)
            one = torch.ones_like(xj)
            I_den = self._trapz_uniform(g_den, one).clamp_min(self.eps)

            # numerator on [0,x_j]
            num_grid = xj * unit
            g_num = self._score_on_grid(j, context, num_grid)
            I_num = self._trapz_uniform(g_num, xj)
            z[:, j] = (I_num / I_den).clamp(0., 1.)

            feat_x = xj if context is None else torch.cat([context, xj], dim=-1)
            g_x = self._positive(self.s_nets[j](feat_x).squeeze(-1)).clamp_min(self.eps)
            log_det = log_det + torch.log(g_x) - torch.log(I_den)
            denoms.append(I_den)

        if return_components:
            return z, log_det, {"denominator": torch.stack(denoms, dim=1)}
        return z, log_det


class HatSpline1D(nn.Module):
    """Piecewise-linear local-support hat basis on [0,1]."""
    def __init__(self, num_knots: int = 16):
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        self.num_knots = num_knots
        self.register_buffer("knots", torch.linspace(0., 1., num_knots))

    def forward(self, x):
        x = x.clamp(0., 1.).unsqueeze(-1)
        delta = (self.knots[1] - self.knots[0]).clamp_min(1e-8)
        return (1. - torch.abs(x - self.knots) / delta).clamp_min(0.)


class SplineExpansion(nn.Module):
    def __init__(self, dim: int, num_knots: int = 16):
        super().__init__()
        self.dim = dim
        self.num_knots = num_knots
        self.bases = nn.ModuleList([HatSpline1D(num_knots) for _ in range(dim)])

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"Expected [B,{self.dim}], got {tuple(z.shape)}")
        return torch.stack([self.bases[j](z[:, j]) for j in range(self.dim)], dim=1)  # [B,d,M]


class AdditiveSplineHead(nn.Module):
    """Strictly additive spline head: output = bias + sum_j g_j(z_j)."""
    def __init__(self, input_dim: int, output_dim: int, num_knots: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_knots = num_knots
        self.spline = SplineExpansion(input_dim, num_knots)
        self.coefficients = nn.Parameter(torch.empty(output_dim, input_dim, num_knots))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.coefficients, mean=0.0, std=0.02)

    def basis(self, z):
        return self.spline(z)

    def feature_effects(self, z):
        phi = self.basis(z)  # [B,d,M]
        # [B,d,O]
        return torch.einsum("bdm,odm->bdo", phi, self.coefficients)

    def forward(self, z):
        effects = self.feature_effects(z)
        return effects.sum(dim=1) + self.bias


class LegacySplineMLPHead(nn.Module):
    """Non-additive spline-MLP head used for the corresponding ablation variant."""
    def __init__(self, input_dim, output_dim, num_knots=16,
                 hidden_dims=(64,64), activation=nn.SiLU, dropout=0.0):
        super().__init__()
        self.spline = SplineExpansion(input_dim, num_knots)
        layers=[]; d=input_dim*num_knots
        for h in hidden_dims:
            layers += [nn.Linear(d,h), activation()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            d=h
        layers.append(nn.Linear(d,output_dim))
        self.net=nn.Sequential(*layers)
    def forward(self,z):
        return self.net(self.spline(z).flatten(1))


class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__(); self.linear=nn.Linear(input_dim, output_dim)
    def forward(self,z): return self.linear(z)


@dataclass
class KRANConfig:
    input_dim: int
    output_dim: int
    task: str = "regression"
    activation: str = "silu"
    mono_hidden_dims: Tuple[int, ...] = (64, 64)
    mono_integrand: str = "softplus"
    mono_num_quad_points: int = 32
    mono_dropout: float = 0.0
    spline_num_knots: int = 16
    head_type: str = "additive"  # additive | legacy_mlp | linear
    head_hidden_dims: Tuple[int, ...] = (64, 64)
    head_dropout: float = 0.0
    identity_transport: bool = False


class KRAN(nn.Module):
    """Knothe-Rosenblatt Additive Network."""
    def __init__(self, config: KRANConfig):
        super().__init__()
        if config.task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification")
        self.config = config
        act = _activation(config.activation)
        self.mono_map = None if config.identity_transport else KRMonoMap(
            config.input_dim, config.mono_hidden_dims, act, config.mono_integrand,
            config.mono_num_quad_points, config.mono_dropout)
        if config.head_type == "additive":
            self.head = AdditiveSplineHead(config.input_dim, config.output_dim, config.spline_num_knots)
        elif config.head_type == "legacy_mlp":
            self.head = LegacySplineMLPHead(config.input_dim, config.output_dim,
                                             config.spline_num_knots, config.head_hidden_dims,
                                             act, config.head_dropout)
        elif config.head_type == "linear":
            self.head = LinearHead(config.input_dim, config.output_dim)
        else:
            raise ValueError("Unknown head_type")

    def forward(self, x, return_extras: bool = False):
        if self.mono_map is None:
            z = x.clamp(0., 1.)
            log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            z, log_det = self.mono_map(x)
        out = self.head(z)
        return (out, {"z": z, "log_det": log_det}) if return_extras else out
