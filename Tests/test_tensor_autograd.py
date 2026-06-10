import numpy as np

from Core.tensor import Tensor


def test_scalar_backward_matches_expected_gradient():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + 2.0 * x

    y.backward()

    assert np.isclose(x.grad, 8.0)


def test_broadcast_add_backward_reduces_gradient_to_bias_shape():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    b = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

    out = (x + b).sum()
    out.backward()

    assert x.grad.shape == (2, 3)
    assert b.grad.shape == (1, 3)
    assert np.allclose(x.grad, np.ones((2, 3)))
    assert np.allclose(b.grad, np.array([[2.0, 2.0, 2.0]]))


def test_matmul_backward_shapes_and_values():
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w_data = np.array([[1.0], [0.5], [-1.0]])

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)

    out = (x @ w).sum()
    out.backward()

    expected_grad_x = np.repeat(w_data.T, repeats=2, axis=0)
    expected_grad_w = x_data.T @ np.ones((2, 1))

    assert x.grad.shape == x_data.shape
    assert w.grad.shape == w_data.shape
    assert np.allclose(x.grad, expected_grad_x)
    assert np.allclose(w.grad, expected_grad_w)
