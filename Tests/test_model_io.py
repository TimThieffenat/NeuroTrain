import json
from pathlib import Path

import numpy as np

from Core.activation import relu
from Core.dropout import Dropout
from Core.layer import Linear
from Core.model import NeuralNetwork
from Core.tensor import Tensor
from Export.model_io import load_model, save_model


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Linear(input_dim, hidden_dim))
    model.add(relu)
    model.add(Linear(hidden_dim, output_dim))
    return model


def test_save_and_load_model_round_trip(tmp_path: Path):
    model = build_model(2, 4, 1)

    x = Tensor(np.array([[0.1, 0.2], [1.0, -1.0], [2.0, 3.0]]))
    y_before = model(x).numpy().copy()

    model_path = tmp_path / "model.json"
    save_model(model, model_path)

    loaded_model = load_model(model_path)
    y_after = loaded_model(x).numpy()

    assert np.allclose(y_before, y_after)


def test_saved_json_contains_layers_and_weights(tmp_path: Path):
    model = build_model(3, 5, 2)
    model_path = tmp_path / "model.json"

    save_model(model, model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))

    assert payload["format"] == "neurotrain-model-v1"
    assert isinstance(payload["layers"], list)
    assert payload["layers"][0]["type"] == "Linear"
    assert "weight" in payload["layers"][0]


def test_save_model_ignores_dropout_layers(tmp_path: Path):
    model = NeuralNetwork()
    model.add(Linear(3, 4))
    model.add(relu)
    model.add(Dropout(0.5))
    model.add(Linear(4, 2))

    model_path = tmp_path / "model_with_dropout.json"
    save_model(model, model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))

    layer_types = [layer["type"] for layer in payload["layers"]]
    assert "Dropout" not in layer_types
    assert layer_types == ["Linear", "Activation", "Linear"]
