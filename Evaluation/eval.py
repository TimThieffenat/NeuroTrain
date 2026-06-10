"""Evaluation helpers separated from training logic."""

from __future__ import annotations

import numpy as np
from typing import Callable

from Data.dataset import Dataset
from Evaluation.metrics import (
    accuracy,
    balanced_accuracy,
    classification_report,
    mae,
    mse,
    precision_recall_f1,
    r2_score,
    regression_report,
    rmse,
    top_k_accuracy,
)


class Tester:
    """Evaluate a trained model on a held-out dataset."""

    __test__ = False

    def __init__(self, model, loss_fn: Callable | None = None):
        self.model = model
        self.loss_fn = loss_fn

    def test_loss(self, dataset: Dataset, batch_size: int = 32) -> float:
        """Compute the average loss on a dataset."""

        if self.loss_fn is None:
            raise ValueError("Tester.test_loss requires a loss_fn")

        losses: list[float] = []
        for x_batch, y_batch in dataset.batches(batch_size=batch_size, shuffle=False):
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_batch, y_pred)
            losses.append(float(loss.numpy()))

        if not losses:
            return float("nan")
        return float(sum(losses) / len(losses))

    def predict(self, dataset: Dataset, batch_size: int = 32):
        """Return stacked true labels and predictions."""

        return predict_dataset(self.model, dataset, batch_size=batch_size)

    def test_classification(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        threshold: float = 0.5,
        average: str = "binary",
    ) -> dict[str, float]:
        """Return classification metrics for the dataset."""

        return evaluate_classification(
            self.model,
            dataset,
            batch_size=batch_size,
            threshold=threshold,
            average=average,
        )

    def test_regression(self, dataset: Dataset, batch_size: int = 32) -> dict[str, float]:
        """Return regression metrics for the dataset."""

        return evaluate_regression(self.model, dataset, batch_size=batch_size)

    def test(
        self,
        dataset: Dataset,
        metrics: str | list[str],
        batch_size: int = 32,
        threshold: float = 0.5,
        average: str = "binary",
    ) -> dict[str, float]:
        """Evaluate one or more metrics on the test dataset."""

        return eval(
            self.model,
            dataset,
            metrics=metrics,
            batch_size=batch_size,
            threshold=threshold,
            average=average,
        )

CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "top_k_accuracy",
}

REGRESSION_METRICS = {
    "mse",
    "rmse",
    "mae",
    "r2",
}


def predict_dataset(model, dataset: Dataset, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked true labels and predictions for a dataset."""

    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    for x_batch, y_batch in dataset.batches(batch_size=batch_size, shuffle=False):
        y_pred = model(x_batch)
        y_true_batches.append(y_batch.numpy())
        y_pred_batches.append(y_pred.numpy())

    if not y_true_batches:
        return np.array([]), np.array([])

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return y_true, y_pred


def evaluate_classification(
    model,
    dataset: Dataset,
    batch_size: int = 32,
    threshold: float = 0.5,
    average: str = "binary",
) -> dict[str, float]:
    """Evaluate classification metrics on a dataset."""

    y_true, y_pred = predict_dataset(model, dataset, batch_size=batch_size)
    if y_true.size == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
        }

    return classification_report(y_true, y_pred, threshold=threshold, average=average)


def evaluate_regression(model, dataset: Dataset, batch_size: int = 32) -> dict[str, float]:
    """Evaluate regression metrics on a dataset."""

    y_true, y_pred = predict_dataset(model, dataset, batch_size=batch_size)
    if y_true.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    return regression_report(y_true, y_pred)


def eval(
    model,
    dataset: Dataset,
    metrics: str | list[str],
    batch_size: int = 32,
    threshold: float = 0.5,
    average: str = "binary",
) -> dict[str, float]:
    """High-level evaluation helper.

    Supported metric names:

    - Classification: accuracy, precision, recall, f1, balanced_accuracy, top_k_accuracy
    - Regression: mse, rmse, mae, r2
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    y_true, y_pred = predict_dataset(model, dataset, batch_size=batch_size)

    if y_true.size == 0:
        return {metric: float("nan") for metric in metrics}

    requested = set(metrics)
    unknown = requested - CLASSIFICATION_METRICS - REGRESSION_METRICS
    if unknown:
        supported = ", ".join(sorted(CLASSIFICATION_METRICS | REGRESSION_METRICS))
        raise ValueError(f"Unknown metric(s): {sorted(unknown)}. Supported metrics: {supported}")

    if requested <= REGRESSION_METRICS:
        report = regression_report(y_true, y_pred)
        return {metric: report[metric] for metric in metrics}

    if requested <= CLASSIFICATION_METRICS:
        precision_score, recall_score, f1_score = precision_recall_f1(
            y_true,
            y_pred,
            threshold=threshold,
            average=average,
        )

        metric_values = {
            "accuracy": accuracy(y_true, y_pred, threshold=threshold),
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "balanced_accuracy": balanced_accuracy(y_true, y_pred, threshold=threshold),
            "top_k_accuracy": top_k_accuracy(y_true, y_pred, k=3)
            if y_pred.ndim == 2 and y_pred.shape[1] > 1
            else float("nan"),
        }

        return {metric: metric_values[metric] for metric in metrics}

    raise ValueError(
        "Cannot mix classification and regression metrics in the same eval() call. "
        "Use either all classification metrics or all regression metrics."
    )


def evaluate(
    model,
    dataset: Dataset,
    metrics: str | list[str],
    batch_size: int = 32,
    threshold: float = 0.5,
    average: str = "binary",
) -> dict[str, float]:
    """Alias for eval()."""

    return eval(
        model=model,
        dataset=dataset,
        metrics=metrics,
        batch_size=batch_size,
        threshold=threshold,
        average=average,
    )
