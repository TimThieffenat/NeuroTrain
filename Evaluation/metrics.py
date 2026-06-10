"""Evaluation metrics for classification and regression tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from Core.tensor import Tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.numpy()
    return np.asarray(value)


def _labels_from_targets(y_true: Any) -> np.ndarray:
    arr = _to_numpy(y_true)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.argmax(arr, axis=1)
    return arr.reshape(-1).astype(int)


def _labels_from_predictions(y_pred: Any, threshold: float = 0.5) -> np.ndarray:
    arr = _to_numpy(y_pred)
    if arr.ndim == 1:
        return (arr >= threshold).astype(int)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return (arr.reshape(-1) >= threshold).astype(int)
    return np.argmax(arr, axis=1)


def confusion_matrix(y_true: Any, y_pred: Any, threshold: float = 0.5) -> np.ndarray:
    """Compute confusion matrix from true labels and model outputs."""

    true_labels = _labels_from_targets(y_true)
    pred_labels = _labels_from_predictions(y_pred, threshold=threshold)

    num_classes = int(max(true_labels.max(initial=0), pred_labels.max(initial=0)) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[true_label, pred_label] += 1
    return matrix


def accuracy(y_true: Any, y_pred: Any, threshold: float = 0.5) -> float:
    true_labels = _labels_from_targets(y_true)
    pred_labels = _labels_from_predictions(y_pred, threshold=threshold)
    return float(np.mean(true_labels == pred_labels))


def _precision_recall_f1_per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1, support


def precision_recall_f1(
    y_true: Any,
    y_pred: Any,
    threshold: float = 0.5,
    average: str = "binary",
) -> tuple[float, float, float]:
    """Return precision, recall and F1 score.

    Supported averages: ``binary``, ``macro``, ``weighted``, ``micro``.
    """

    cm = confusion_matrix(y_true, y_pred, threshold=threshold)
    precision_c, recall_c, f1_c, support_c = _precision_recall_f1_per_class(cm)

    if average == "binary":
        if len(precision_c) < 2:
            idx = 0
        else:
            idx = 1
        return float(precision_c[idx]), float(recall_c[idx]), float(f1_c[idx])

    if average == "macro":
        return float(np.mean(precision_c)), float(np.mean(recall_c)), float(np.mean(f1_c))

    if average == "weighted":
        total = support_c.sum()
        if total == 0:
            return 0.0, 0.0, 0.0
        return (
            float(np.sum(precision_c * support_c) / total),
            float(np.sum(recall_c * support_c) / total),
            float(np.sum(f1_c * support_c) / total),
        )

    if average == "micro":
        tp_total = float(np.trace(cm))
        total = float(cm.sum())
        if total == 0:
            return 0.0, 0.0, 0.0
        score = tp_total / total
        return score, score, score

    raise ValueError("average must be one of: binary, macro, weighted, micro")


def balanced_accuracy(y_true: Any, y_pred: Any, threshold: float = 0.5) -> float:
    cm = confusion_matrix(y_true, y_pred, threshold=threshold)
    _, recall_c, _, _ = _precision_recall_f1_per_class(cm)
    return float(np.mean(recall_c))


def specificity(y_true: Any, y_pred: Any, threshold: float = 0.5) -> float:
    """Compute specificity (true negative rate) for binary classification."""

    cm = confusion_matrix(y_true, y_pred, threshold=threshold)
    if cm.shape != (2, 2):
        raise ValueError("specificity is only defined for binary classification")

    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def top_k_accuracy(y_true: Any, y_pred: Any, k: int = 3) -> float:
    """Compute Top-K accuracy for multi-class prediction scores."""

    true_labels = _labels_from_targets(y_true)
    scores = _to_numpy(y_pred)
    if scores.ndim != 2:
        raise ValueError("top_k_accuracy expects y_pred with shape (n_samples, n_classes)")
    if k <= 0:
        raise ValueError("k must be > 0")

    k = min(k, scores.shape[1])
    topk_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    hits = [true_label in row for true_label, row in zip(true_labels, topk_idx)]
    return float(np.mean(hits))


def classification_report(
    y_true: Any,
    y_pred: Any,
    threshold: float = 0.5,
    average: str = "binary",
) -> dict[str, float]:
    """Aggregate common classification metrics in one dictionary."""

    precision_score, recall_score, f1_score = precision_recall_f1(
        y_true,
        y_pred,
        threshold=threshold,
        average=average,
    )

    report = {
        "accuracy": accuracy(y_true, y_pred, threshold=threshold),
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, threshold=threshold),
    }

    cm = confusion_matrix(y_true, y_pred, threshold=threshold)
    if cm.shape == (2, 2):
        report["specificity"] = specificity(y_true, y_pred, threshold=threshold)
    return report


def mse(y_true: Any, y_pred: Any) -> float:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.mean((pred - true) ** 2))


def rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Any, y_pred: Any) -> float:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(pred - true)))


def r2_score(y_true: Any, y_pred: Any) -> float:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)

    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def regression_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Aggregate common regression metrics in one dictionary."""

    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
