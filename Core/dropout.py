"""Dropout regularization layer."""

import numpy as np

from Core.layer import Layer
from Core.tensor import Tensor


class Dropout(Layer):
	"""Dropout regularization layer.
	
	During training, randomly sets activations to zero with probability `p`.
	During inference, scales remaining activations by `1/(1-p)` (inverted dropout).
	
	This prevents co-adaptation of neurons and reduces overfitting.
	"""

	def __init__(self, p: float = 0.5):
		"""Initialize dropout layer.
		
		Parameters
		----------
		p : float
			Probability of dropping a neuron. Default: 0.5
			Valid range: [0, 1)
		"""
		if not 0 <= p < 1:
			raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
		
		self.p = p
		self.training = True  # Default to training mode
		self.mask = None  # Store mask for backprop
	
	def set_training_mode(self, training: bool) -> None:
		"""Set dropout to training or inference mode.
		
		Parameters
		----------
		training : bool
			If True, dropout is active. If False, dropout is disabled.
		"""
		self.training = training
	
	def train(self) -> "Dropout":
		"""Set to training mode (dropout active)."""
		self.training = True
		return self
	
	def eval(self) -> "Dropout":
		"""Set to evaluation mode (dropout inactive)."""
		self.training = False
		return self
	
	def forward(self, x: Tensor) -> Tensor:
		"""Apply dropout to input.
		
		Parameters
		----------
		x : Tensor
			Input tensor of any shape.
		
		Returns
		-------
		Tensor
			Output tensor with same shape as input.
		"""
		if not self.training or self.p == 0:
			# During inference or if p=0, pass through unchanged
			return x
		
		# Generate binary mask: 1 with probability (1-p), 0 with probability p
		keep_prob = 1.0 - self.p
		self.mask = np.random.binomial(1, keep_prob, size=x.shape)
		
		# Apply mask and scale by 1/keep_prob (inverted dropout)
		scale = 1.0 / keep_prob
		dropped_data = x.data * self.mask * scale
		
		# Create output tensor with backprop support
		out = Tensor(
			dropped_data,
			copy=False,
			requires_grad=x.requires_grad,
			_children=(x,),
			_op="dropout",
		)
		
		def _backward() -> None:
			if out.grad is None:
				return
			
			# Backprop: apply same mask and scaling
			if x.grad is None:
				x.grad = np.zeros_like(x.data)
			x.grad += out.grad * self.mask * scale
		
		out._backward = _backward
		return out
