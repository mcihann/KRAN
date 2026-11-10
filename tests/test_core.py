import torch
from kran.model import KRAN, KRANConfig


def test_shapes_and_bounds():
    m=KRAN(KRANConfig(4,2,task='classification',mono_num_quad_points=8,spline_num_knots=8))
    x=torch.rand(6,4); y,e=m(x,return_extras=True)
    assert y.shape==(6,2); assert e['z'].shape==(6,4); assert torch.all(e['z']>=0) and torch.all(e['z']<=1)


def test_additive_decomposition():
    m=KRAN(KRANConfig(3,1)); x=torch.rand(5,3); z,_=m.mono_map(x)
    eff=m.head.feature_effects(z)
    assert torch.allclose(m.head(z),eff.sum(1)+m.head.bias,atol=1e-6)


def test_monotone_coordinate():
    m=KRAN(KRANConfig(3,1,mono_num_quad_points=16)); x=torch.rand(10,3)*.8
    z,_=m.mono_map(x); xp=x.clone(); xp[:,1]+=1e-3; zp,_=m.mono_map(xp)
    assert torch.all(zp[:,1] >= z[:,1]-1e-6)


def test_transport_is_triangular_under_late_feature_change():
    torch.manual_seed(8)
    m = KRAN(KRANConfig(4, 2, task='classification', mono_num_quad_points=8))
    x = torch.rand(6, 4)
    changed = x.clone()
    changed[:, 2] = 1.0 - changed[:, 2]
    z0, _ = m.mono_map(x)
    z1, _ = m.mono_map(changed)
    assert torch.allclose(z0[:, :2], z1[:, :2], atol=1e-7)


def test_normalized_transport_boundary_values():
    torch.manual_seed(9)
    m = KRAN(KRANConfig(3, 1, mono_num_quad_points=16))
    x = torch.rand(4, 3)
    for j in range(3):
        at_zero = x.clone()
        at_one = x.clone()
        at_zero[:, j] = 0.0
        at_one[:, j] = 1.0
        z0, _ = m.mono_map(at_zero)
        z1, _ = m.mono_map(at_one)
        assert torch.allclose(z0[:, j], torch.zeros(4), atol=1e-6)
        assert torch.allclose(z1[:, j], torch.ones(4), atol=1e-6)


def test_multioutput_regression_additive_head():
    torch.manual_seed(10)
    m = KRAN(KRANConfig(3, 2, task='regression', spline_num_knots=8))
    x = torch.rand(5, 3)
    predictions, details = m(x, return_extras=True)
    components = m.head.feature_effects(details['z'])
    assert predictions.shape == (5, 2)
    assert components.shape == (5, 3, 2)
    assert torch.allclose(predictions, components.sum(1) + m.head.bias, atol=1e-6)
