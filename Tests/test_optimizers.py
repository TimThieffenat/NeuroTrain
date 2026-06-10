import numpy as np

from Core.optimizer import Adam, SGD
from Core.tensor import Tensor


def test_sgd_step_updates_parameter():
    param = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    param.grad = np.array([0.5, -0.5])

    optimizer = SGD([param], lr=0.1)
    optimizer.step()

    assert np.allclose(param.data, np.array([0.95, -1.95]))


def test_optimizer_zero_grad_clears_gradients():
    param = Tensor(np.array([1.0]), requires_grad=True)
    param.grad = np.array([1.23])

    optimizer = SGD([param], lr=0.1)
    optimizer.zero_grad()

    assert param.grad is None


def test_adam_step_updates_parameter():
    param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    optimizer = Adam([param], lr=0.01)

    param.grad = np.array([0.1, -0.2])
    before = param.data.copy()
    optimizer.step()

    assert np.any(before != param.data)
