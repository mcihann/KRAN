"""Data loading, leakage-aware preprocessing, nested holdout splits, and training helpers.

Evaluation is defined by explicit seeds and settings. Dataset versions, installed
dependencies, and execution backends must be recorded to compare independent runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (fetch_california_housing, fetch_covtype,
                              fetch_openml, load_breast_cancer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


@dataclass
class RawDataset:
    name: str
    task: str
    X: Any
    y: np.ndarray
    batch_size: int
    feature_names: list[str]


PRESET = {
    'breast_cancer': ('Breast Cancer', 'classification', 32),
    'covertype': ('Covertype', 'classification', 256),
    'adult_income': ('Adult Income', 'classification', 256),
    'energy_efficiency': ('Energy Efficiency', 'regression', 32),
    'california_housing': ('California Housing', 'regression', 256),
    'yearpredictionmsd': ('YearPredictionMSD', 'regression', 256),
}


def load_raw_dataset(name: str) -> RawDataset:
    """Load public data; fixed subsets are formed *before* any outer split."""
    if name not in PRESET:
        raise ValueError(f'Unsupported dataset: {name}')
    label, task, batch_size = PRESET[name]
    if name == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        names = list(data.feature_names)
    elif name == 'covertype':
        data = fetch_covtype()
        indices = np.random.RandomState(42).choice(len(data.data), 50000, replace=False)
        X = data.data[indices].astype(np.float32)
        y = data.target[indices].astype(np.int64) - 1
        names = [f'feature_{i}' for i in range(X.shape[1])]
    elif name == 'adult_income':
        data = fetch_openml('adult', version=2, as_frame=True)
        X = data.data.copy()
        y = data.target.astype(str).str.contains('>50K', regex=False).astype(np.int64).to_numpy()
        names = list(X.columns)
    elif name == 'energy_efficiency':
        data = fetch_openml('energy_efficiency', as_frame=True)
        X = data.data.copy()
        y = np.asarray(data.target, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        # Some OpenML fetch configurations expose only one default target.
        # Require the expected two-output heating/cooling target definition.
        if y.shape[1] != 2:
            target_cols = [c for c in ('Y1', 'Y2') if c in X.columns]
            if len(target_cols) == 2:
                y = X[target_cols].to_numpy(dtype=np.float32)
                X = X.drop(columns=target_cols)
            else:
                raise ValueError('Energy Efficiency must include two targets (heating and cooling load). '
                                 'Pin/check your OpenML dataset version and target columns.')
        X = X.to_numpy(dtype=np.float32)
        names = [f'feature_{i}' for i in range(X.shape[1])]
    elif name == 'california_housing':
        data = fetch_california_housing()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)[:, None]
        names = list(data.feature_names)
    else:
        data = fetch_openml(data_id=44027, as_frame=False)
        X = np.asarray(data.data[:50000], dtype=np.float32)
        y = np.asarray(data.target[:50000], dtype=np.float32).reshape(-1, 1)
        names = [f'feature_{i}' for i in range(X.shape[1])]
    expected = {'breast_cancer': (569, 30), 'covertype': (50000, 54),
                'adult_income': (48842, 14), 'energy_efficiency': (768, 8),
                'california_housing': (20640, 8), 'yearpredictionmsd': (50000, 90)}
    observed = (len(X), X.shape[1])
    if observed != expected[name]:
        raise ValueError(f'Dataset {name} has {observed} samples/features; expected '
                         f'{expected[name]}. Check dataset version before claiming reproducibility.')
    return RawDataset(label, task, X, y, batch_size, names)


def row_select(X: Any, indices: np.ndarray) -> Any:
    return X.iloc[indices].copy() if hasattr(X, 'iloc') else X[indices]


def split_indices(y: np.ndarray, task: str, seed: int, test_size: float = 0.2):
    idx = np.arange(len(y))
    stratify = y if task == 'classification' else None
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)


def nested_indices(raw: RawDataset, seed: int):
    outer_train, outer_test = split_indices(raw.y, raw.task, seed)
    inner_local_train, inner_local_val = split_indices(
        raw.y[outer_train], raw.task, seed + 10000,
    )
    return outer_train, outer_test, outer_train[inner_local_train], outer_train[inner_local_val]


class TrainingPreprocessor:
    """Fit one-hot encoding and MinMax exclusively on supplied training rows."""

    def fit(self, X: Any):
        if isinstance(X, pd.DataFrame):
            cats = list(X.select_dtypes(exclude=np.number).columns)
            nums = [c for c in X.columns if c not in cats]
            self.encoder = ColumnTransformer(
                [('num', 'passthrough', nums),
                 ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)],
                sparse_threshold=0,
            ) if cats else None
            features = self.encoder.fit_transform(X) if self.encoder is not None else X.to_numpy()
            self.feature_names = list(self.encoder.get_feature_names_out()) if self.encoder is not None else nums
        else:
            self.encoder = None
            features = np.asarray(X)
            self.feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        self.scaler = MinMaxScaler(clip=True).fit(np.asarray(features, dtype=np.float32))
        return self

    def transform(self, X: Any) -> np.ndarray:
        features = self.encoder.transform(X) if self.encoder is not None else np.asarray(X)
        return self.scaler.transform(np.asarray(features, dtype=np.float32)).astype(np.float32)

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)


def fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """SHA256 of shape, dtype and contiguous bytes; independent of CSV rounding."""
    h = hashlib.sha256()
    for obj in (X, y):
        a = np.ascontiguousarray(obj)
        h.update(str((a.shape, str(a.dtype))).encode('utf-8'))
        h.update(a.tobytes())
    return h.hexdigest()


def output_dim(y: np.ndarray, task: str) -> int:
    if task == 'classification':
        return int(np.unique(y).size)
    return y.shape[1] if y.ndim > 1 else 1


def target_statistics(y: np.ndarray):
    arr = np.asarray(y, dtype=np.float32).reshape(len(y), -1)
    mean = arr.mean(axis=0)
    scale = arr.std(axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


def metrics_and_predictions(model, X: np.ndarray, y: np.ndarray, task: str,
                            device: str, batch_size: int, target_stats=None):
    import torch
    from .metrics import classification_metrics, regression_metrics
    from .train import predict
    from torch.utils.data import DataLoader, TensorDataset
    tensor_y = torch.long if task == 'classification' else torch.float32
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                      torch.tensor(y, dtype=tensor_y)),
                        batch_size=batch_size, shuffle=False)
    y_true, pred, infer_time = predict(model, loader, task, device=device)
    if target_stats is not None:
        mean, scale = target_stats
        pred = pred * scale + mean
    metrics = classification_metrics(y_true, pred) if task == 'classification' else regression_metrics(y_true, pred)
    return metrics, pred, infer_time


def select_neural_epochs(model, Xtr, ytr, Xval, yval, task, batch_size,
                         max_epochs: int, device: str, seed: int, min_delta: float = 1e-6,
                         patience: int = 20):
    """Min validation NLL/MSE; regression validation on original target scale."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .train import set_seed
    from .metrics import regression_metrics, classification_metrics

    set_seed(seed)
    model.to(device)
    stats = target_statistics(ytr) if task == 'regression' else None
    yfit = (ytr - stats[0]) / stats[1] if stats is not None else ytr
    ds = TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32),
                       torch.as_tensor(yfit, dtype=torch.long if task == 'classification' else torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = float('inf')
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.nn.functional.cross_entropy(pred, yb) if task == 'classification' else torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        val_preds = []
        with torch.no_grad():
            for start in range(0, len(Xval), batch_size):
                xb = torch.as_tensor(Xval[start:start + batch_size], dtype=torch.float32, device=device)
                val_preds.append(model(xb).cpu().numpy())
        pred = np.concatenate(val_preds)
        if task == 'classification':
            from scipy.special import softmax
            from sklearn.metrics import log_loss
            score = float(log_loss(yval, softmax(pred, axis=1), labels=list(range(pred.shape[1]))))
        else:
            from sklearn.metrics import mean_squared_error
            score = float(mean_squared_error(yval, pred * stats[1] + stats[0]))
        if score < best_score - min_delta:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch, best_score


def fit_full_neural(model, Xtr, ytr, task, batch_size, epochs, device, seed):
    import torch
    from .train import loaders, train_model, set_seed
    set_seed(seed)
    stats = target_statistics(ytr) if task == 'regression' else None
    yfit = (ytr - stats[0]) / stats[1] if stats is not None else ytr
    tr, _ = loaders(Xtr, yfit, Xtr[:1], yfit[:1], batch_size, task)
    _, seconds = train_model(model, tr, task, epochs, device=device,
                             lr=1e-3, weight_decay=1e-4)
    return seconds, stats


def tree_estimator(name, task, n_classes, y, seed, candidate):
    """Explicit, limited compact/capacity candidate definitions."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    multi = task == 'regression' and np.asarray(y).ndim == 2 and np.asarray(y).shape[1] > 1
    if name == 'RandomForest':
        return (RandomForestClassifier if task == 'classification' else RandomForestRegressor)(
            n_estimators=300, random_state=seed, n_jobs=-1)
    if name == 'HistGradientBoosting':
        base = (HistGradientBoostingClassifier if task == 'classification' else HistGradientBoostingRegressor)(random_state=seed)
        return MultiOutputRegressor(base) if multi else base
    n_est, depth, rate = (300, 4, .05) if candidate == 'compact' else (600, 6, .035)
    if name == 'XGBoost':
        import xgboost as xgb
        params = dict(n_estimators=n_est, max_depth=depth, learning_rate=rate,
                      subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                      min_child_weight=1, tree_method='hist', random_state=seed,
                      n_jobs=4)
        base = xgb.XGBClassifier(objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
                                  **params) if task == 'classification' else xgb.XGBRegressor(objective='reg:squarederror', **params)
    elif name == 'LightGBM':
        import lightgbm as lgb
        params = dict(n_estimators=n_est, max_depth=depth, learning_rate=rate,
                      num_leaves=15 if candidate == 'compact' else 31,
                      subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                      random_state=seed, deterministic=True, force_col_wise=True,
                      n_jobs=4, verbosity=-1)
        base = lgb.LGBMClassifier(**params) if task == 'classification' else lgb.LGBMRegressor(**params)
    elif name == 'CatBoost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(iterations=n_est, depth=depth, learning_rate=rate,
                      l2_leaf_reg=3, random_seed=seed, verbose=False,
                      thread_count=4, task_type='CPU')
        base = CatBoostClassifier(loss_function='Logloss' if n_classes == 2 else 'MultiClass', **params) if task == 'classification' else CatBoostRegressor(loss_function='RMSE', **params)
    else:
        raise ValueError(name)
    return MultiOutputRegressor(base) if multi else base


def select_tree(name, raw, X_inner, X_val, y_inner, y_val, seed):
    from sklearn.metrics import log_loss, mean_squared_error
    candidates = ('compact', 'capacity') if name in ('XGBoost', 'LightGBM', 'CatBoost') else ('fixed',)
    scores = []
    for candidate in candidates:
        model = tree_estimator(name, raw.task, output_dim(raw.y, raw.task), y_inner, seed, candidate)
        model.fit(X_inner, y_inner.ravel() if raw.task == 'regression' and output_dim(raw.y, raw.task) == 1 else y_inner)
        if raw.task == 'classification':
            p = model.predict_proba(X_val)
            score = log_loss(y_val, p, labels=list(range(p.shape[1])))
        else:
            score = mean_squared_error(y_val, model.predict(X_val))
        scores.append((score, candidate))
    return min(scores)[1], float(min(scores)[0])
