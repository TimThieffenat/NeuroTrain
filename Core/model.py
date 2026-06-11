"""High-level model container with a simple API."""

from typing import Callable, Iterable

from Core.layer import Activation, Layer
from Core.tensor import Tensor


class Model(Layer):
    """Composable model container.

    This class allows building networks step by step:

    - ``model.add(layer)`` for regular layers
    - ``model.add(activation_function)`` for callables such as ``relu``
    """

    def __init__(self, layers: Iterable[Layer] | None = None):
        self.layers = list(layers or [])

    def add(self, block: Layer | Callable[[Tensor], Tensor], name: str | None = None) -> "Model":
        """Append a layer or activation block to the model."""

        if isinstance(block, Layer):
            self.layers.append(block)
            return self

        if callable(block):
            activation_name = name or getattr(block, "__name__", "activation")
            self.layers.append(Activation(function=block, name=activation_name))
            return self

        raise TypeError("block must be a Layer instance or a callable activation")

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

    def train(self) -> "Model":
        """Set model to training mode (dropout active)."""

        for layer in self.layers:
            if hasattr(layer, "set_training_mode"):
                layer.set_training_mode(True)
        return self

    def eval(self) -> "Model":
        """Set model to evaluation mode (dropout inactive)."""

        for layer in self.layers:
            if hasattr(layer, "set_training_mode"):
                layer.set_training_mode(False)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """Inference with dropout disabled."""

        self.eval()
        return self(x)
