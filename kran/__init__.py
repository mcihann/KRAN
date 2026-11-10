"""KRAN: Knothe-Rosenblatt Additive Network for interpretable tabular learning."""
from .model import KRAN, KRANConfig, KRMonoMap, AdditiveSplineHead, HatSpline1D, SplineExpansion

__all__ = [
    "KRAN", "KRANConfig", "KRMonoMap", "AdditiveSplineHead", "HatSpline1D", "SplineExpansion"
]
__version__ = "1.1.0"
