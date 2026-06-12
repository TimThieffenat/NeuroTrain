"""Model abstractions and neural-network implementation."""

from abc import ABC, abstractmethod
from typing import Callable, Iterable

from Core.layer import Activation, Layer
from Core.tensor import Tensor


class Model(ABC):
    """Common contract for any trainable model in the framework."""

    def set_training_mode(self, training: bool) -> None:
        """Hook for model families that need train/eval behavior.

        Default implementation is a no-op (useful for tree-based estimators).
        """

        _ = training

    def train(self) -> "Model":
        """Set model to training mode when applicable."""

        self.set_training_mode(True)
        return self

    def eval(self) -> "Model":
        """Set model to evaluation mode when applicable."""

        self.set_training_mode(False)
        return self

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Run inference and return predictions."""


class TreeModel(Model, ABC):
    """Base class for tree-based models (Decision Tree, RF, XGBoost-style).

    This class defines a common non-neural API so every tree estimator can
    plug into evaluation and project scripts with a consistent interface.
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: Tensor, y: Tensor) -> "TreeModel":
        """Train the estimator using input features and targets."""

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Return model predictions for each sample."""

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities when available.

        Tree classifiers should override this method. Regressors can keep the
        default behavior and only implement ``predict``.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba"
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has already been trained."""

        return self._is_fitted

    def _mark_fitted(self) -> None:
        """Mark estimator as trained (to be called by subclasses)."""

        self._is_fitted = True


class NeuralNetwork(Model, Layer):
    """Composable neural network container.

    This class allows building networks step by step:

    - ``model.add(layer)`` for regular layers
    - ``model.add(activation_function)`` for callables such as ``relu``
    """

    def __init__(self, layers: Iterable[Layer] | None = None):
        self.layers = list(layers or [])

    def add(self, block: Layer | Callable[[Tensor], Tensor], name: str | None = None) -> "NeuralNetwork":
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

    def set_training_mode(self, training: bool) -> None:
        """Toggle training/eval mode for layers that support it."""

        for layer in self.layers:
            if hasattr(layer, "set_training_mode"):
                layer.set_training_mode(training)

    def predict(self, x: Tensor) -> Tensor:
        """Inference with dropout disabled."""

        self.eval()
        return self(x)


# Backward-compatible alias during transition.
NeuralNetworkModel = NeuralNetwork
LegacyModel = NeuralNetwork
