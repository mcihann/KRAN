import numpy as np
import torch

from kran.attributions import integrated_gradients, permutation_nll
from kran.explanations import original_feature_responses
from kran.model import KRAN, KRANConfig
from kran.repeated import RawDataset, TrainingPreprocessor, nested_indices, fingerprint


def test_original_feature_decomposition_includes_downstream_and_last_is_zero():
    torch.manual_seed(42)
    model = KRAN(KRANConfig(3, 2, task='classification',
                                 mono_num_quad_points=8, spline_num_knots=8))
    x = np.array([[.25, .4, .55], [.7, .2, .8]], dtype=np.float32)
    grid = np.linspace(.1, .9, 5, dtype=np.float32)
    for j in range(3):
        response = original_feature_responses(model, x, j, grid)
        assert response['decomposition_max_error'] < 1e-5
        assert np.allclose(response['direct'] + response['downstream'], response['total'], atol=1e-5)
        if j == 2:
            assert np.array_equal(response['downstream'], np.zeros_like(response['downstream']))
        assert response['probability_change_class0'].shape == (len(x), len(grid))


def test_integrated_gradients_linear_binary_margin_complete():
    model = torch.nn.Linear(3, 2)
    x = np.array([[.1, .2, .3], [.9, .4, .5]], dtype=np.float32)
    baseline = np.zeros(3, dtype=np.float32)
    attribution, residual = integrated_gradients(model, x, baseline, steps=32, chunk_size=8)
    weights = (model.weight[0] - model.weight[1]).detach().numpy()
    assert np.allclose(attribution, x * weights, atol=2e-6)
    assert np.max(np.abs(residual)) < 2e-6


def test_train_only_preprocessing_clips_unseen_test_extremes():
    train = np.array([[0, 2], [2, 4], [3, 5]], dtype=np.float32)
    test = np.array([[30, 200]], dtype=np.float32)
    p = TrainingPreprocessor().fit(train)
    expected = p.transform(train)
    assert np.all(expected <= 1) and np.all(expected >= 0)
    assert np.array_equal(p.transform(test), np.ones((1, 2), dtype=np.float32))
    assert fingerprint(expected, np.array([1, 2, 3])) == fingerprint(expected.copy(), np.array([1, 2, 3]))
    assert fingerprint(expected, np.array([1, 2, 3])) != fingerprint(expected, np.array([1, 2, 4]))


def test_nested_indices_disjoint_and_80_20_ratios():
    raw = RawDataset('synthetic', 'classification', np.random.default_rng(42).normal(size=(100, 3)),
                     np.tile([0, 1], 50), 16, ['a', 'b', 'c'])
    outer_train, outer_test, inner_train, inner_val = nested_indices(raw, 1042)
    assert (len(outer_train), len(outer_test), len(inner_train), len(inner_val)) == (80, 20, 64, 16)
    assert set(outer_train).isdisjoint(outer_test)
    assert set(inner_train).isdisjoint(inner_val)
    assert set(inner_train).union(inner_val) == set(outer_train)


def test_reference_config_parameter_count():
    config = KRANConfig(30, 2, task='classification', spline_num_knots=16)
    model = KRAN(config)
    assert sum(p.numel() for p in model.parameters()) == 159392
