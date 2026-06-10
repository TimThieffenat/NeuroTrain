"""Loss functions for the neural network framework.
"""

from Core.tensor import Tensor

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


def categorical_cross_entropy(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
	"""Categorical cross-entropy for one-hot targets.

	``y_true`` should contain one-hot encoded labels and ``y_pred`` should
	contain class probabilities for each sample.
	"""

	true_tensor = y_true
	pred_tensor = y_pred.clip(eps, 1.0 - eps)
	per_sample = -(true_tensor * pred_tensor.log()).sum(axis=-1)
	return per_sample.mean()
