import numpy as np

from Core.activation import relu
from Core.layer import Linear
from Core.model import NeuralNetwork
from Core.loss import mse_loss
from Core.optimizer import SGD
from Core.tensor import Tensor
from Data.dataset import Dataset, train_val_split
from Evaluation.eval import eval
from Train.train import train


def build_model() -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Linear(2, 4))
    model.add(relu)
    model.add(Linear(4, 1))
    return model


def test_train_function_runs_and_returns_history():
    np.random.seed(0)
    x_np = np.random.randn(60, 2)
    y_np = (2.0 * x_np[:, :1] - x_np[:, 1:2])

    x = Tensor(x_np)
    y = Tensor(y_np)
    ds = Dataset(x, y)
    train_ds, val_ds = train_val_split(ds, val_ratio=0.2, shuffle=True)

    model = build_model()
    optimizer = SGD(model.parameters(), lr=0.01)

    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=mse_loss,
        optimizer=optimizer,
        epochs=2,
        batch_size=8,
        verbose=False,
    )

    assert len(history) == 2
    assert "train_loss" in history[-1]
    assert "val_loss" in history[-1]


def test_eval_function_returns_requested_metrics():
    np.random.seed(1)
    x_np = np.random.randn(40, 2)
    y_np = (2.0 * x_np[:, :1] - x_np[:, 1:2])

    x = Tensor(x_np)
    y = Tensor(y_np)
    ds = Dataset(x, y)
    _, val_ds = train_val_split(ds, val_ratio=0.25, shuffle=False)

    model = build_model()
    metrics = eval(model=model, dataset=val_ds, metrics=["mse", "rmse", "mae", "r2"])

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}
    assert all(np.isfinite(v) for v in metrics.values())
