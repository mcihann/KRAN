"""Self-contained KRAN classification example using scikit-learn Breast Cancer data.

Run from the repository root after installing the project:
    python -m examples.train_breast_cancer --epochs 5
"""
import argparse

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from kran import KRAN, KRANConfig
from kran.metrics import classification_metrics
from kran.train import loaders, predict, set_seed, train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but not available')

    if args.device == 'cpu':
        # Avoid excessive thread overhead for the small demonstration dataset.
        torch.set_num_threads(1)
    set_seed(42)
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2,
        stratify=dataset.target, random_state=42)

    # Fit preprocessing ONLY on training samples.
    scaler = MinMaxScaler(clip=True).fit(X_train)
    X_train = scaler.transform(X_train).astype('float32')
    X_test = scaler.transform(X_test).astype('float32')
    train_loader, test_loader = loaders(
        X_train, y_train, X_test, y_test, batch_size=32, task='classification')

    model = KRAN(KRANConfig(
        input_dim=X_train.shape[1], output_dim=2, task='classification',
        mono_integrand='softplus', mono_num_quad_points=32,
        spline_num_knots=16, head_type='additive'))
    _, elapsed = train_model(
        model, train_loader, 'classification', epochs=args.epochs, device=args.device)
    true_labels, probabilities, _ = predict(
        model, test_loader, 'classification', device=args.device)
    scores = classification_metrics(true_labels, probabilities)
    print(f'Device: {args.device}; epochs: {args.epochs}; train time (s): {elapsed:.2f}')
    for metric, value in scores.items():
        print(f'{metric}: {value:.4f}')
    print('Illustrative train/test split only; not a nested-holdout evaluation.')


if __name__ == '__main__':
    main()
