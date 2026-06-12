import numpy as np

from Core.activation import relu
from Core.layer import Linear
from Core.model import NeuralNetwork
from Core.tensor import Tensor


def test_model_add_layer_and_activation_callable():
    model = NeuralNetwork()
    model.add(Linear(2, 3)).add(relu).add(Linear(3, 1))

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = model(x)

    assert y.shape == (2, 1)
    assert len(model.layers) == 3


def test_model_parameters_collect_linear_weights_and_biases():
    model = NeuralNetwork()
    model.add(Linear(2, 4)).add(relu).add(Linear(4, 1))

    params = model.parameters()

    # 2 Linear layers => (weight,bias) * 2
    assert len(params) == 4
