import numpy as np

from Core.loss import binary_cross_entropy, mse_loss
from Core.tensor import Tensor


def test_mse_loss_value():
    y_true = Tensor(np.array([[1.0], [3.0]]))
    y_pred = Tensor(np.array([[2.0], [1.0]]))

    loss = mse_loss(y_true, y_pred)

    # ((1-2)^2 + (3-1)^2) / 2 = (1 + 4) / 2 = 2.5
    assert np.isclose(float(loss.numpy()), 2.5)


def test_binary_cross_entropy_backward_populates_gradients():
    y_true = Tensor(np.array([[1.0], [0.0], [1.0]]))
    y_pred = Tensor(np.array([[0.9], [0.2], [0.8]]), requires_grad=True)

    loss = binary_cross_entropy(y_true, y_pred)
    loss.backward()

    assert y_pred.grad is not None
    assert y_pred.grad.shape == y_pred.shape
    assert np.all(np.isfinite(y_pred.grad))
