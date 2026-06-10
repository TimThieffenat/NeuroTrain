"""Optimizers for the neural network framework."""

from abc import ABC
from typing import Iterable
import numpy as np

from Core.tensor import Tensor


class Optimizer(ABC):
    """Base class for all optimizers.

    Parameters
    ----------
    parameters:
        Iterable of trainable tensors.
    """

    def __init__(self, parameters: Iterable[Tensor]):
        self.parameters = list(parameters)

    def zero_grad(self) -> None:
        """Clear gradients stored on every parameter."""

        for parameter in self.parameters:
            if hasattr(parameter, "zero_grad"):
                parameter.zero_grad()
            else:
                parameter.grad = None

    def step(self) -> None:
        """Update parameters in place."""

        raise NotImplementedError

    @staticmethod
    def _get_grad(parameter: Tensor) -> np.ndarray | None:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            return None
        if isinstance(grad, Tensor):
            return grad.numpy()
        return np.asarray(grad)


class SGD(Optimizer):
    """Stochastic gradient descent.

    Parameters
    ----------
    parameters:
        Trainable tensors.
    lr:
        Learning rate.
    momentum:
        Optional momentum factor. Set to ``0.0`` to disable momentum.
    """

    def __init__(self, parameters: Iterable[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for parameter in self.parameters:
            grad = self._get_grad(parameter)
            if grad is None:
                continue

            if self.momentum:
                velocity = self._velocity.get(id(parameter))
                if velocity is None:
                    velocity = np.zeros_like(parameter.data)
                velocity = self.momentum * velocity + grad
                self._velocity[id(parameter)] = velocity
                update = velocity
            else:
                update = grad

            parameter.data = parameter.data - self.lr * update


class Adam(Optimizer):
    """Adam optimizer with the standard update rule.

    Parameters
    ----------
    parameters:
        Trainable tensors.
    lr:
        Learning rate.
    beta1:
        Exponential decay for the first moment.
    beta2:
        Exponential decay for the second moment.
    eps:
        Small constant for numerical stability.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._step_count = 0
        self._first_moment: dict[int, np.ndarray] = {}
        self._second_moment: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self._step_count += 1

        for parameter in self.parameters:
            grad = self._get_grad(parameter)
            if grad is None:
                continue

            parameter_id = id(parameter)
            first_moment = self._first_moment.get(parameter_id, np.zeros_like(parameter.data))
            second_moment = self._second_moment.get(parameter_id, np.zeros_like(parameter.data))

            first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * grad
            second_moment = self.beta2 * second_moment + (1.0 - self.beta2) * (grad * grad)

            first_moment_hat = first_moment / (1.0 - self.beta1 ** self._step_count)
            second_moment_hat = second_moment / (1.0 - self.beta2 ** self._step_count)

            parameter.data = parameter.data - self.lr * first_moment_hat / (np.sqrt(second_moment_hat) + self.eps)

            self._first_moment[parameter_id] = first_moment
            self._second_moment[parameter_id] = second_moment