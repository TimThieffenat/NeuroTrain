import numpy as np

from Core.activation import relu
from Core.layer import Linear
from Core.model import NeuralNetwork
from Core.loss import mse_loss
from Core.tensor import Tensor
from Data.dataset import Dataset, train_val_split
from Evaluation.eval import Tester


def build_model() -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Linear(2, 4))
    model.add(relu)
    model.add(Linear(4, 1))
    return model


def test_tester_can_compute_loss_and_metrics():
    np.random.seed(2)
    x_np = np.random.randn(30, 2)
    y_np = (2.0 * x_np[:, :1] - x_np[:, 1:2])

    x = Tensor(x_np)
    y = Tensor(y_np)
    ds = Dataset(x, y)
    _, test_ds = train_val_split(ds, val_ratio=0.2, shuffle=False)

    model = build_model()
    tester = Tester(model=model, loss_fn=mse_loss)

    loss = tester.test_loss(test_ds)
    metrics = tester.test(test_ds, metrics=["mse", "rmse", "mae", "r2"])

    assert np.isfinite(loss)
    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}
    assert all(np.isfinite(v) for v in metrics.values())
