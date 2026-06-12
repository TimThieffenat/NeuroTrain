"""Training loop utilities with autograd support."""

from typing import Callable
from pathlib import Path

import numpy as np

from Data.dataset import Dataset
from Evaluation.metrics import accuracy, precision_recall_f1, balanced_accuracy


class NeuralNetworkTrainer:
    """Trainer for neural network models.

    Parameters
    ----------
    model:
        Callable object that takes inputs and returns predictions.
    loss_fn:
        Function ``loss_fn(y_true, y_pred)`` returning a scalar tensor.
        If loss_fn supports class_weights, they will be passed automatically.
    optimizer:
        Optimizer instance. Optional if you only want loss evaluation.
    class_weights : np.ndarray, optional
        Class weights for imbalanced datasets. Shape: (n_classes,)
        Passed to loss_fn if it accepts a class_weights parameter.
    """

    def __init__(self, model, loss_fn: Callable, optimizer=None, class_weights: np.ndarray | None = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.class_weights = class_weights
        self.history: list[dict[str, float]] = []
        self.metrics_history: dict[str, list[float]] = {}

    def _call_loss_fn(self, y_true, y_pred):
        """Call loss function with class_weights if supported."""
        if self.class_weights is not None:
            try:
                return self.loss_fn(y_true, y_pred, class_weights=self.class_weights)
            except TypeError:
                # If loss_fn doesn't support class_weights, call without them
                return self.loss_fn(y_true, y_pred)
        return self.loss_fn(y_true, y_pred)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: bool = True,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        save_metrics: list[str] | None = None,
    ) -> list[dict[str, float]]:
        """Run training and return per-epoch loss history.

        When ``early_stopping`` is enabled, training stops when validation loss
        does not improve by at least ``min_delta`` for ``patience`` consecutive
        epochs. If ``restore_best_weights`` is True, model parameters are reset
        to the best validation-loss checkpoint before returning.
        """

        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        tracked_metrics = list(save_metrics or ["train_loss", "val_loss"])
        self.metrics_history = {metric_name: [] for metric_name in tracked_metrics}
        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_params: list[np.ndarray] | None = None
        epochs_without_improvement = 0

        def snapshot_parameters() -> list[np.ndarray]:
            return [np.array(param.data, copy=True) for param in self.model.parameters()]

        def restore_parameters(params_snapshot: list[np.ndarray]) -> None:
            for param, saved in zip(self.model.parameters(), params_snapshot):
                param.data[...] = saved

        for epoch in range(1, epochs + 1):
            train_losses: list[float] = []
            self.model.train()  # Enable dropout during training

            for x_batch, y_batch in train_dataset.batches(batch_size=batch_size, shuffle=shuffle):
                y_pred = self.model(x_batch)
                loss = self._call_loss_fn(y_batch, y_pred)
                train_losses.append(float(loss.numpy()))

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            self.model.eval()  # Disable dropout during validation
            val_loss = self.evaluate_loss(val_dataset, batch_size=batch_size)

            epoch_metrics: dict[str, float] = {"epoch": float(epoch)}
            full_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                **self._compute_requested_metrics(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    batch_size=batch_size,
                    requested=tracked_metrics,
                ),
            }

            for metric_name, metric_value in full_metrics.items():
                if metric_name in tracked_metrics:
                    epoch_metrics[metric_name] = metric_value
                    self.metrics_history[metric_name].append(metric_value)

            history.append(epoch_metrics)

            if verbose:
                metric_chunks: list[str] = []
                for metric_name in tracked_metrics:
                    if metric_name in epoch_metrics:
                        metric_chunks.append(f"{metric_name}: {epoch_metrics[metric_name]:.6f}")
                if not metric_chunks:
                    metric_chunks = [f"train_loss: {train_loss:.6f}", f"val_loss: {val_loss:.6f}"]
                print(f"Epoch {epoch}/{epochs} - " + " - ".join(metric_chunks))

            improved = val_loss < (best_val_loss - min_delta)
            if improved:
                best_val_loss = val_loss
                best_params = snapshot_parameters()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best_val_loss={best_val_loss:.6f})"
                    )
                break

        if restore_best_weights and best_params is not None:
            restore_parameters(best_params)

        self.history = history
        return history

    def _predict_dataset(self, dataset: Dataset, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        y_true_batches: list[np.ndarray] = []
        y_pred_batches: list[np.ndarray] = []
        self.model.eval()  # Disable dropout during prediction

        for x_batch, y_batch in dataset.batches(batch_size=batch_size, shuffle=False):
            y_pred = self.model(x_batch)
            y_true_batches.append(y_batch.numpy())
            y_pred_batches.append(y_pred.numpy())

        if not y_true_batches:
            return np.array([]), np.array([])

        return np.vstack(y_true_batches), np.vstack(y_pred_batches)

    def _compute_split_metrics(self, dataset: Dataset, batch_size: int) -> dict[str, float]:
        y_true, y_pred = self._predict_dataset(dataset, batch_size=batch_size)
        if y_true.size == 0:
            return {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "balanced_accuracy": float("nan"),
            }

        precision_score, recall_score, f1_score = precision_recall_f1(
            y_true,
            y_pred,
            average="macro",
        )

        return {
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        }

    def _compute_requested_metrics(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        requested: list[str],
    ) -> dict[str, float]:
        supported = {"accuracy", "precision", "recall", "f1", "balanced_accuracy"}
        need_train = any(name.startswith("train_") and name[6:] in supported for name in requested)
        need_val = any(name.startswith("val_") and name[4:] in supported for name in requested)

        computed: dict[str, float] = {}

        if need_train:
            train_scores = self._compute_split_metrics(train_dataset, batch_size=batch_size)
            for metric_name, metric_value in train_scores.items():
                computed[f"train_{metric_name}"] = metric_value

        if need_val:
            val_scores = self._compute_split_metrics(val_dataset, batch_size=batch_size)
            for metric_name, metric_value in val_scores.items():
                computed[f"val_{metric_name}"] = metric_value

        return computed

    def plot_history(self, metric_name: str, title: str, save_path: str | Path) -> Path:
        """Save a training history plot to disk without displaying it."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plot_history") from exc

        output_path = Path(save_path)
        if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = output_path.parent / f"{metric_name}_history.png"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if metric_name in {"loss", "accuracy", "precision", "recall", "f1", "balanced_accuracy"}:
            train_key = f"train_{metric_name}"
            val_key = f"val_{metric_name}"
            series = [
                (train_key, self.metrics_history.get(train_key, []), "Train"),
                (val_key, self.metrics_history.get(val_key, []), "Validation"),
            ]
        else:
            series = [(metric_name, self.metrics_history.get(metric_name, []), metric_name)]

        if not any(values for _, values, _ in series):
            raise ValueError(f"No history found for metric '{metric_name}'")

        plt.figure(figsize=(8, 5))
        for _, values, label in series:
            if values:
                epochs = np.arange(1, len(values) + 1)
                plt.plot(epochs, values, label=label)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    def evaluate_loss(self, dataset: Dataset, batch_size: int = 32) -> float:
        """Compute average loss on a dataset."""

        losses: list[float] = []
        self.model.eval()  # Disable dropout during evaluation
        for x_batch, y_batch in dataset.batches(batch_size=batch_size, shuffle=False):
            y_pred = self.model(x_batch)
            loss = self._call_loss_fn(y_batch, y_pred)
            losses.append(float(loss.numpy()))

        if not losses:
            return float("nan")
        return float(np.mean(losses))

    # Backward-compatible alias.
    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> float:
        return self.evaluate_loss(dataset, batch_size=batch_size)


def train(
    model,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn: Callable,
    optimizer=None,
    epochs: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    verbose: bool = True,
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    save_metrics: list[str] | None = None,
    class_weights: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """High-level training helper using the NeuralNetworkTrainer class."""

    trainer = NeuralNetworkTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer, class_weights=class_weights)
    return trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=verbose,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        save_metrics=save_metrics,
    )


# Backward-compatible alias during transition.
Trainer = NeuralNetworkTrainer
