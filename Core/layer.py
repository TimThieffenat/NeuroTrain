"""Base layers used to build simple neural networks."""

from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np

from Core.tensor import Tensor


class Layer(ABC):
    """Base class for every layer in the framework.

    The idea is simple: a layer receives a tensor and returns a tensor.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute the output of the layer."""

    def __call__(self, x: Tensor) -> Tensor:
        """Allow the layer to be called like a function."""

        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return trainable parameters, if the layer has any."""

        return []


class Linear(Layer):
    """Fully connected layer: ``y = x @ W + b``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, size=(in_features, out_features)),
            requires_grad=True,
        )
        self.bias = Tensor.zeros(out_features, requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self) -> list[Tensor]:
        parameters = [self.weight]
        if self.bias is not None:
            parameters.append(self.bias)
        return parameters


class Activation(Layer):
    """Wrap a pure activation function in the layer interface."""

    def __init__(self, function: Callable[[Tensor], Tensor], name: str = "activation"):
        self.function = function
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        return self.function(x)


class Sequential(Layer):
    """Apply layers one after another."""

    def __init__(self, layers: Iterable[Layer]):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


