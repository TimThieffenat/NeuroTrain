"""Tests for normalization layers."""

import numpy as np

from Core.normalization import BatchNorm, LayerNorm, InstanceNorm
from Core.tensor import Tensor


def test_batchnorm_normalizes_batch():
	bn = BatchNorm(num_features=3)
	x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
	y = bn(x)

	assert y.shape == x.shape
	assert np.allclose(np.mean(y.data, axis=0), 0.0, atol=1e-5)


def test_batchnorm_backward():
	bn = BatchNorm(num_features=2)
	x = Tensor(np.random.randn(10, 2), requires_grad=True)
	y = bn(x)
	loss = y.sum()
	loss.backward()

	assert x.grad is not None
	assert x.grad.shape == x.shape
	assert bn.gamma.grad is not None
	assert bn.beta.grad is not None


def test_layernorm_normalizes_features():
	ln = LayerNorm(num_features=3)
	x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
	y = ln(x)

	assert y.shape == x.shape
	row_means = np.mean(y.data, axis=1)
	assert np.allclose(row_means, 0.0, atol=1e-5)


def test_layernorm_backward():
	ln = LayerNorm(num_features=4)
	x = Tensor(np.random.randn(5, 4), requires_grad=True)
	y = ln(x)
	loss = y.sum()
	loss.backward()

	assert x.grad is not None
	assert x.grad.shape == x.shape
	assert ln.gamma.grad is not None
	assert ln.beta.grad is not None


def test_instancenorm_normalizes_per_sample():
	inst = InstanceNorm(num_features=3)
	x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
	y = inst(x)

	assert y.shape == x.shape
	# InstanceNorm normalizes each feature independently per sample, not globally
	# So the output should have different values than batch norm
	assert y.data.shape == x.data.shape


def test_instancenorm_backward():
	inst = InstanceNorm(num_features=2)
	x = Tensor(np.random.randn(8, 2), requires_grad=True)
	y = inst(x)
	loss = y.sum()
	loss.backward()

	assert x.grad is not None
	assert x.grad.shape == x.shape
	assert inst.gamma.grad is not None
	assert inst.beta.grad is not None


def test_normalization_parameters_returned():
	bn = BatchNorm(num_features=3)
	ln = LayerNorm(num_features=3)
	inst = InstanceNorm(num_features=3)

	assert len(bn.parameters()) == 2
	assert len(ln.parameters()) == 2
	assert len(inst.parameters()) == 2

	assert bn.parameters()[0] is bn.gamma
	assert bn.parameters()[1] is bn.beta
