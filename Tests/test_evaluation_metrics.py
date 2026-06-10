import numpy as np

from Evaluation.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    precision_recall_f1,
    regression_report,
    top_k_accuracy,
)


def test_confusion_matrix_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.7, 0.6])

    cm = confusion_matrix(y_true, y_pred, threshold=0.5)

    assert cm.shape == (2, 2)
    assert np.array_equal(cm, np.array([[1, 1], [0, 2]]))


def test_classification_report_binary_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.7, 0.6])

    report = classification_report(y_true, y_pred, threshold=0.5, average="binary")

    assert np.isclose(report["accuracy"], 0.75)
    assert np.isclose(report["precision"], 2 / 3)
    assert np.isclose(report["recall"], 1.0)
    assert np.isclose(report["f1"], 0.8)
    assert "specificity" in report


def test_precision_recall_f1_macro_multiclass():
    y_true = np.array([0, 1, 2, 1])
    y_pred_scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
        ]
    )

    precision, recall, f1 = precision_recall_f1(y_true, y_pred_scores, average="macro")

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_top_k_accuracy_multiclass():
    y_true = np.array([0, 1, 2])
    y_pred_scores = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.4, 0.3, 0.3],
            [0.2, 0.5, 0.3],
        ]
    )

    top1 = top_k_accuracy(y_true, y_pred_scores, k=1)
    top2 = top_k_accuracy(y_true, y_pred_scores, k=2)

    assert np.isclose(top1, accuracy(y_true, y_pred_scores))
    assert top2 >= top1


def test_regression_report_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    report = regression_report(y_true, y_pred)

    assert set(report.keys()) == {"mse", "rmse", "mae", "r2"}
    assert report["mse"] >= 0.0
    assert report["rmse"] >= 0.0
    assert report["mae"] >= 0.0
