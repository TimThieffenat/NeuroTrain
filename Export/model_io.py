"""Model serialization utilities (JSON) for NeuroTrain."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from Core.activation import relu, sigmoid, softmax, tanh
from Core.layer import Activation, Layer, Linear
from Core.model import Model
from Core.normalization import BatchNorm, LayerNorm, InstanceNorm
from Core.tensor import Tensor

_FORMAT_VERSION = "neurotrain-model-v1"

_ACTIVATION_REGISTRY = {
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "tanh": tanh,
}


def _serialize_layer(layer: Layer) -> dict[str, Any]:
    if isinstance(layer, Linear):
        payload: dict[str, Any] = {
            "type": "Linear",
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "use_bias": layer.bias is not None,
            "weight": layer.weight.numpy().tolist(),
        }
        if layer.bias is not None:
            payload["bias"] = layer.bias.numpy().tolist()
        return payload

    if isinstance(layer, Activation):
        return {
            "type": "Activation",
            "name": layer.name,
        }

    if isinstance(layer, BatchNorm):
        return {
            "type": "BatchNorm",
            "num_features": layer.num_features,
            "eps": layer.eps,
            "momentum": layer.momentum,
            "gamma": layer.gamma.numpy().tolist(),
            "beta": layer.beta.numpy().tolist(),
            "running_mean": layer.running_mean.tolist(),
            "running_var": layer.running_var.tolist(),
        }

    if isinstance(layer, LayerNorm):
        return {
            "type": "LayerNorm",
            "num_features": layer.num_features,
            "eps": layer.eps,
            "gamma": layer.gamma.numpy().tolist(),
            "beta": layer.beta.numpy().tolist(),
        }

    if isinstance(layer, InstanceNorm):
        return {
            "type": "InstanceNorm",
            "num_features": layer.num_features,
            "eps": layer.eps,
            "gamma": layer.gamma.numpy().tolist(),
            "beta": layer.beta.numpy().tolist(),
        }

    raise ValueError(f"Unsupported layer type for serialization: {type(layer).__name__}")


def _deserialize_layer(payload: dict[str, Any]) -> Layer:
    layer_type = payload.get("type")

    if layer_type == "Linear":
        layer = Linear(
            in_features=int(payload["in_features"]),
            out_features=int(payload["out_features"]),
            bias=bool(payload.get("use_bias", True)),
        )
        layer.weight = Tensor(np.asarray(payload["weight"], dtype=float), requires_grad=True)

        if payload.get("use_bias", True):
            layer.bias = Tensor(np.asarray(payload["bias"], dtype=float), requires_grad=True)
        else:
            layer.bias = None
        return layer

    if layer_type == "Activation":
        name = str(payload["name"])
        function = _ACTIVATION_REGISTRY.get(name)
        if function is None:
            supported = ", ".join(sorted(_ACTIVATION_REGISTRY))
            raise ValueError(
                f"Unsupported activation '{name}' in JSON. Supported activations: {supported}"
            )
        return Activation(function=function, name=name)

    if layer_type == "BatchNorm":
        layer = BatchNorm(
            num_features=int(payload["num_features"]),
            eps=float(payload.get("eps", 1e-5)),
            momentum=float(payload.get("momentum", 0.9)),
        )
        layer.gamma = Tensor(np.asarray(payload["gamma"], dtype=float), requires_grad=True)
        layer.beta = Tensor(np.asarray(payload["beta"], dtype=float), requires_grad=True)
        layer.running_mean = np.asarray(payload["running_mean"], dtype=float)
        layer.running_var = np.asarray(payload["running_var"], dtype=float)
        return layer

    if layer_type == "LayerNorm":
        layer = LayerNorm(
            num_features=int(payload["num_features"]),
            eps=float(payload.get("eps", 1e-5)),
        )
        layer.gamma = Tensor(np.asarray(payload["gamma"], dtype=float), requires_grad=True)
        layer.beta = Tensor(np.asarray(payload["beta"], dtype=float), requires_grad=True)
        return layer

    if layer_type == "InstanceNorm":
        layer = InstanceNorm(
            num_features=int(payload["num_features"]),
            eps=float(payload.get("eps", 1e-5)),
        )
        layer.gamma = Tensor(np.asarray(payload["gamma"], dtype=float), requires_grad=True)
        layer.beta = Tensor(np.asarray(payload["beta"], dtype=float), requires_grad=True)
        return layer

    raise ValueError(f"Unsupported layer type in JSON: {layer_type}")


def save_model(model: Model, file_path: str | Path) -> Path:
    """Save a model architecture and weights to JSON.

    The model must expose a ``layers`` attribute compatible with ``Model``.
    """

    if not hasattr(model, "layers"):
        raise ValueError("Model must expose a 'layers' attribute to be serialized")

    layers_payload = [_serialize_layer(layer) for layer in model.layers]
    payload = {
        "format": _FORMAT_VERSION,
        "model_class": model.__class__.__name__,
        "layers": layers_payload,
    }

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_model(file_path: str | Path) -> Model:
    """Load a model from JSON and return a ``Model`` instance."""

    input_path = Path(file_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    if payload.get("format") != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model format: {payload.get('format')}. Expected '{_FORMAT_VERSION}'."
        )

    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, list):
        raise ValueError("Invalid model JSON: 'layers' must be a list")

    layers = [_deserialize_layer(item) for item in layers_payload]
    return Model(layers)



def save_predictions(
    test_csv_path: str | Path,
    predictions: np.ndarray,
    output_path: str | Path,
    class_names: list[str] | None = None,
    id_column: int = 0,
    delimiter: str = ",",
    skip_header: bool = False,
) -> None:
    """
    Save predictions with IDs to CSV.

    Extracts IDs from the test CSV, decodes one-hot predictions to class strings,
    and saves as CSV with columns: id, class.

    Args:
        test_csv_path: Path to test CSV (to extract IDs)
        predictions: Predictions array with shape (n_samples, n_classes)
        output_path: Path where to save the results CSV
        class_names: List of class names (from training dataset.class_names).
                    If None, uses 'class_0', 'class_1', ...
        id_column: Column index containing IDs (default 0)
        delimiter: CSV delimiter
        skip_header: Whether to skip the first row of test CSV (header)
    """
    # Load test CSV to extract IDs
    test_data = np.genfromtxt(
        test_csv_path, delimiter=delimiter, dtype=str, skip_header=int(skip_header)
    )

    if test_data.ndim == 1:
        test_data = test_data.reshape(-1, 1)

    # Extract IDs from specified column
    ids = test_data[:, id_column]

    # Decode predictions: take argmax to get class indices
    class_indices = np.argmax(predictions, axis=1)

    # Use provided class names or generate default ones
    if class_names is None:
        class_names = [f"class_{i}" for i in range(predictions.shape[1])]

    # Map indices to class names
    predicted_classes = [class_names[int(idx)] for idx in class_indices]

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results to CSV
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"id{delimiter}class\n")
        for id_val, pred_class in zip(ids, predicted_classes):
            f.write(f"{id_val}{delimiter}{pred_class}\n")
