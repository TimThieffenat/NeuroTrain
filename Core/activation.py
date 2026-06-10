"""Small collection of differentiable activation functions."""

import numpy as np

from Core.tensor import Tensor


def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0.0, self.data),
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out


def sigmoid(self) -> "Tensor":
        # Stable sigmoid: avoid overflow for large negative/positive inputs.
        sig = np.empty_like(self.data, dtype=np.float64)
        positive_mask = self.data >= 0
        negative_mask = ~positive_mask

        sig[positive_mask] = 1.0 / (1.0 + np.exp(-self.data[positive_mask]))
        exp_x = np.exp(self.data[negative_mask])
        sig[negative_mask] = exp_x / (1.0 + exp_x)

        out = Tensor(sig, copy=False, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * sig * (1.0 - sig))

        out._backward = _backward
        return out


def tanh(self) -> "Tensor":
        tanh_value = np.tanh(self.data)
        out = Tensor(tanh_value, copy=False, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (1.0 - tanh_value * tanh_value))

        out._backward = _backward
        return out

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Return softmax probabilities along ``axis``."""

    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)

    out = Tensor(
        probs,
        copy=False,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="softmax",
    )

    def _backward() -> None:
        if out.grad is None or not x.requires_grad:
            return

        # Jacobian-vector product for softmax without building full Jacobian.
        dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
        x._accumulate_grad(probs * (out.grad - dot))

    out._backward = _backward
    return out