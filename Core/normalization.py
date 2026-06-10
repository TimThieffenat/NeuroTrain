"""Normalization layers with learnable parameters."""

import numpy as np

from Core.layer import Layer
from Core.tensor import Tensor


class BatchNorm(Layer):
	"""Batch normalization layer.
	
	Normalizes activations per feature across the batch dimension.
	Parameters: gamma (scale) and beta (shift).
	"""

	def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum

		self.gamma = Tensor(np.ones(num_features), requires_grad=True)
		self.beta = Tensor(np.zeros(num_features), requires_grad=True)

		self.running_mean = np.zeros(num_features)
		self.running_var = np.ones(num_features)

	def forward(self, x: Tensor) -> Tensor:
		"""Normalize and scale/shift the input.
		
		x: shape (batch_size, num_features)
		output: shape (batch_size, num_features)
		"""

		if x.shape[-1] != self.num_features:
			raise ValueError(
				f"Expected input with {self.num_features} features, "
				f"got {x.shape[-1]}"
			)

		batch_mean = np.mean(x.data, axis=0, keepdims=True)
		batch_var = np.var(x.data, axis=0, keepdims=True)

		self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean.squeeze()
		self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * batch_var.squeeze()

		x_normalized = (x.data - batch_mean) / np.sqrt(batch_var + self.eps)

		out_data = self.gamma.data * x_normalized + self.beta.data
		out = Tensor(
			out_data,
			copy=False,
			requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
			_children=(x, self.gamma, self.beta),
			_op="batchnorm",
		)

		def _backward() -> None:
			if out.grad is None:
				return

			N = x.data.shape[0]

			if self.gamma.requires_grad:
				gamma_grad = np.sum(out.grad * x_normalized, axis=0)
				self.gamma._accumulate_grad(gamma_grad)

			if self.beta.requires_grad:
				beta_grad = np.sum(out.grad, axis=0)
				self.beta._accumulate_grad(beta_grad)

			if x.requires_grad:
				dL_dx_normalized = out.grad * self.gamma.data
				dvar = np.sum(
					dL_dx_normalized * (x.data - batch_mean) * -0.5 * (batch_var + self.eps) ** -1.5,
					axis=0,
					keepdims=True,
				)
				dmean = np.sum(dL_dx_normalized * -1.0 / np.sqrt(batch_var + self.eps), axis=0, keepdims=True)
				dmean = dmean + dvar * np.sum(-2.0 * (x.data - batch_mean), axis=0, keepdims=True) / N

				x_grad = (dL_dx_normalized / np.sqrt(batch_var + self.eps)) + (dvar * 2.0 * (x.data - batch_mean) / N) + (dmean / N)
				x._accumulate_grad(x_grad)

		out._backward = _backward
		return out

	def parameters(self) -> list[Tensor]:
		return [self.gamma, self.beta]


class LayerNorm(Layer):
	"""Layer normalization layer.
	
	Normalizes activations per sample, across the feature dimension.
	Parameters: gamma (scale) and beta (shift).
	"""

	def __init__(self, num_features: int, eps: float = 1e-5):
		self.num_features = num_features
		self.eps = eps

		self.gamma = Tensor(np.ones(num_features), requires_grad=True)
		self.beta = Tensor(np.zeros(num_features), requires_grad=True)

	def forward(self, x: Tensor) -> Tensor:
		"""Normalize and scale/shift the input.
		
		x: shape (..., num_features)
		output: shape (..., num_features)
		"""

		if x.shape[-1] != self.num_features:
			raise ValueError(
				f"Expected input with {self.num_features} features, "
				f"got {x.shape[-1]}"
			)

		mean = np.mean(x.data, axis=-1, keepdims=True)
		var = np.var(x.data, axis=-1, keepdims=True)

		x_normalized = (x.data - mean) / np.sqrt(var + self.eps)

		out_data = self.gamma.data * x_normalized + self.beta.data
		out = Tensor(
			out_data,
			copy=False,
			requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
			_children=(x, self.gamma, self.beta),
			_op="layernorm",
		)

		def _backward() -> None:
			if out.grad is None:
				return

			if self.gamma.requires_grad:
				gamma_grad = np.sum(out.grad * x_normalized, axis=tuple(range(out.grad.ndim - 1)))
				self.gamma._accumulate_grad(gamma_grad)

			if self.beta.requires_grad:
				beta_grad = np.sum(out.grad, axis=tuple(range(out.grad.ndim - 1)))
				self.beta._accumulate_grad(beta_grad)

			if x.requires_grad:
				dL_dx_normalized = out.grad * self.gamma.data
				N = x.data.shape[-1]

				dvar = np.sum(
					dL_dx_normalized * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5,
					axis=-1,
					keepdims=True,
				)
				dmean = np.sum(dL_dx_normalized * -1.0 / np.sqrt(var + self.eps), axis=-1, keepdims=True)
				dmean = dmean + dvar * np.sum(-2.0 * (x.data - mean), axis=-1, keepdims=True) / N

				x_grad = (dL_dx_normalized / np.sqrt(var + self.eps)) + (dvar * 2.0 * (x.data - mean) / N) + (dmean / N)
				x._accumulate_grad(x_grad)

		out._backward = _backward
		return out

	def parameters(self) -> list[Tensor]:
		return [self.gamma, self.beta]


class InstanceNorm(Layer):
	"""Instance normalization layer.
	
	Normalizes activations per sample and per feature independently.
	Used primarily in style transfer and image tasks.
	Parameters: gamma (scale) and beta (shift).
	"""

	def __init__(self, num_features: int, eps: float = 1e-5):
		self.num_features = num_features
		self.eps = eps

		self.gamma = Tensor(np.ones(num_features), requires_grad=True)
		self.beta = Tensor(np.zeros(num_features), requires_grad=True)

	def forward(self, x: Tensor) -> Tensor:
		"""Normalize and scale/shift the input.
		
		x: shape (batch_size, num_features) or (batch_size, num_features, ...)
		output: same shape as input
		"""

		if x.shape[1] != self.num_features:
			raise ValueError(
				f"Expected input with {self.num_features} features, "
				f"got {x.shape[1]}"
			)

		axes_to_reduce = tuple(range(2, x.ndim)) if x.ndim > 2 else ()

		if x.ndim == 2:
			mean = x.data
			var = np.var(x.data, axis=0, keepdims=True)
			mean = np.mean(x.data, axis=0, keepdims=True)
		else:
			mean = np.mean(x.data, axis=axes_to_reduce, keepdims=True)
			var = np.var(x.data, axis=axes_to_reduce, keepdims=True)

		x_normalized = (x.data - mean) / np.sqrt(var + self.eps)

		if x.ndim == 2:
			out_data = self.gamma.data * x_normalized + self.beta.data
		else:
			gamma_reshaped = self.gamma.data.reshape(1, -1, *([1] * (x.ndim - 2)))
			beta_reshaped = self.beta.data.reshape(1, -1, *([1] * (x.ndim - 2)))
			out_data = gamma_reshaped * x_normalized + beta_reshaped

		out = Tensor(
			out_data,
			copy=False,
			requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
			_children=(x, self.gamma, self.beta),
			_op="instancenorm",
		)

		def _backward() -> None:
			if out.grad is None:
				return

			if self.gamma.requires_grad:
				if x.ndim == 2:
					gamma_grad = np.sum(out.grad * x_normalized, axis=0)
				else:
					gamma_grad = np.sum(out.grad * x_normalized, axis=tuple(range(out.grad.ndim)))
					gamma_grad = np.reshape(gamma_grad, self.gamma.shape)
				self.gamma._accumulate_grad(gamma_grad)

			if self.beta.requires_grad:
				if x.ndim == 2:
					beta_grad = np.sum(out.grad, axis=0)
				else:
					beta_grad = np.sum(out.grad, axis=tuple(range(out.grad.ndim)))
					beta_grad = np.reshape(beta_grad, self.beta.shape)
				self.beta._accumulate_grad(beta_grad)

			if x.requires_grad:
				if x.ndim == 2:
					gamma_data = self.gamma.data
				else:
					gamma_data = self.gamma.data.reshape(1, -1, *([1] * (x.ndim - 2)))

				dL_dx_normalized = out.grad * gamma_data

				if x.ndim == 2:
					N = x.data.shape[0]
				else:
					N = np.prod(x.data.shape[2:])

				dvar = np.sum(
					dL_dx_normalized * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5,
					axis=axes_to_reduce if axes_to_reduce else (0,),
					keepdims=True,
				)
				dmean = np.sum(dL_dx_normalized * -1.0 / np.sqrt(var + self.eps), axis=axes_to_reduce if axes_to_reduce else (0,), keepdims=True)
				dmean = dmean + dvar * np.sum(-2.0 * (x.data - mean), axis=axes_to_reduce if axes_to_reduce else (0,), keepdims=True) / N

				x_grad = (dL_dx_normalized / np.sqrt(var + self.eps)) + (dvar * 2.0 * (x.data - mean) / N) + (dmean / N)
				x._accumulate_grad(x_grad)

		out._backward = _backward
		return out

	def parameters(self) -> list[Tensor]:
		return [self.gamma, self.beta]
