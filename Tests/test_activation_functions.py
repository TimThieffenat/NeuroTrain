import numpy as np

from Core.activation import softmax
from Core.tensor import Tensor


def test_softmax_outputs_probabilities_per_row():
    x = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 3.0, 3.0]]))
    y = softmax(x, axis=1)

    assert y.shape == (2, 3)
    row_sums = y.numpy().sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(2), rtol=1e-7, atol=1e-7)
    assert np.all(y.numpy() >= 0.0)
    assert np.all(y.numpy() <= 1.0)


def test_softmax_backward_produces_finite_gradient():
    x = Tensor(np.array([[1.0, 0.5, -0.25]]), requires_grad=True)
    target = Tensor(np.array([[0.0, 1.0, 0.0]]))

    y = softmax(x, axis=1)
    loss = (y * target).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert np.all(np.isfinite(x.grad))
