"""Loss functions for the neural network framework.
"""

import numpy as np
from Core.tensor import Tensor


def compute_class_weights(y_true: Tensor | np.ndarray, method: str = "balanced") -> np.ndarray:
	"""Compute class weights for imbalanced datasets.
	
	Parameters
	----------
	y_true : Tensor or np.ndarray
		One-hot encoded targets of shape (n_samples, n_classes)
	method : str, default="balanced"
		Method for computing weights:
		- "balanced": weight = n_samples / (n_classes * n_class_samples)
		  Automatically accounts for class imbalance
		- "inverse_freq": weight = 1 / frequency
		  Simpler, inversely proportional to class frequency
	
	Returns
	-------
	np.ndarray
		Class weights of shape (n_classes,)
	
	Example
	-------
	>>> y = Tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
	>>> weights = compute_class_weights(y)
	>>> weights
	array([0.5, 1.0, 2.0])  # Class 0 oversampled, class 2 undersampled
	"""
	
	if isinstance(y_true, Tensor):
		y_data = y_true.data
	else:
		y_data = np.asarray(y_true)
	
	if y_data.ndim != 2:
		raise ValueError(f"Expected 2D one-hot encoded targets, got shape {y_data.shape}")
	
	n_samples, n_classes = y_data.shape
	class_counts = np.sum(y_data, axis=0)
	
	if method == "balanced":
		# weight = total / (n_classes * count_per_class)
		# This ensures equal total weight per class
		weights = n_samples / (n_classes * class_counts + 1e-10)
	elif method == "inverse_freq":
		# weight = 1 / frequency
		weights = 1.0 / (class_counts + 1e-10)
	else:
		raise ValueError(f"Unknown method: {method}. Use 'balanced' or 'inverse_freq'")
	
	return weights.astype(np.float32)

def mse_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
	"""Mean squared error.

	This is the classic regression loss
	"""

	diff = y_pred - y_true
	return (diff * diff).mean()


def mae_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
	"""Mean absolute error.

	This loss is less sensitive to large outliers than MSE.
	"""

	return (y_pred - y_true).abs().mean()


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
	"""Binary cross-entropy for targets in ``{0, 1}``.

	``y_pred`` is expected to contain probabilities between 0 and 1. The
	``eps`` value avoids ``log(0)``.
	"""

	true_tensor = y_true
	pred_tensor = y_pred.clip(eps, 1.0 - eps)
	loss = -(true_tensor * pred_tensor.log() + (1.0 - true_tensor) * (1.0 - pred_tensor).log())
	return loss.mean()


def categorical_cross_entropy(
	y_true: Tensor, 
	y_pred: Tensor, 
	eps: float = 1e-7, 
	class_weights: np.ndarray | None = None
) -> Tensor:
	"""Categorical cross-entropy for one-hot targets.

	``y_true`` should contain one-hot encoded labels and ``y_pred`` should
	contain class probabilities for each sample.
	
	Parameters
	----------
	y_true : Tensor
		One-hot encoded targets
	y_pred : Tensor
		Predicted probabilities
	eps : float
		Small value to avoid log(0)
	class_weights : np.ndarray, optional
		Weights per class of shape (n_classes,). If provided, loss is weighted
		to account for class imbalance. Computed via compute_class_weights().
	"""
	
	true_tensor = y_true
	pred_tensor = y_pred.clip(eps, 1.0 - eps)
	per_sample = -(true_tensor * pred_tensor.log()).sum(axis=-1)
	
	if class_weights is not None:
		# Get class index for each sample (argmax of one-hot)
		class_indices = np.argmax(y_true.data, axis=1)
		# Apply weights: each sample's loss is multiplied by its class weight
		sample_weights = class_weights[class_indices]
		weighted_loss = per_sample * Tensor(sample_weights, requires_grad=False)
		return weighted_loss.mean()
	
	return per_sample.mean()
