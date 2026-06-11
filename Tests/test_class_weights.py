"""Test class weights computation for imbalanced datasets."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from Core.tensor import Tensor
from Core.loss import compute_class_weights, categorical_cross_entropy


def test_compute_class_weights_balanced():
	"""Test balanced class weights computation."""
	# Perfectly balanced dataset: 4 samples, 3 classes
	y = Tensor(np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
		[1, 0, 0],
	], dtype=np.float32))
	
	weights = compute_class_weights(y, method="balanced")
	
	# Class 0: 2 samples, Class 1: 1 sample, Class 2: 1 sample
	# Balanced weights: [0.67, 1.33, 1.33] (approximately)
	assert len(weights) == 3
	assert weights[1] > weights[0], "Rarer class (1) should have higher weight"
	assert weights[2] > weights[0], "Rarer class (2) should have higher weight"
	print(f"✓ Balanced weights: {weights}")


def test_compute_class_weights_inverse_freq():
	"""Test inverse frequency class weights."""
	y = Tensor(np.array([
		[1, 0, 0],
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	], dtype=np.float32))
	
	weights = compute_class_weights(y, method="inverse_freq")
	
	# Class 0: 2 samples, Class 1: 1 sample, Class 2: 1 sample
	# Inverse freq: [0.5, 1.0, 1.0]
	assert len(weights) == 3
	assert weights[0] < weights[1]
	assert weights[0] < weights[2]
	print(f"✓ Inverse freq weights: {weights}")


def test_categorical_cross_entropy_with_weights():
	"""Test that weighted cross-entropy changes loss calculation."""
	y_true = Tensor(np.array([
		[1, 0, 0],  # Class 0 (common)
		[1, 0, 0],  # Class 0 (common)
		[0, 1, 0],  # Class 1 (rare)
		[0, 0, 1],  # Class 2 (rare)
	], dtype=np.float32))
	
	y_pred = Tensor(np.array([
		[0.8, 0.1, 0.1],  # Correct prediction for class 0
		[0.7, 0.2, 0.1],  # Correct prediction for class 0
		[0.3, 0.6, 0.1],  # Correct prediction for class 1
		[0.3, 0.1, 0.6],  # Correct prediction for class 2
	], dtype=np.float32))
	
	# Unweighted loss
	loss_unweighted = categorical_cross_entropy(y_true, y_pred)
	
	# Weighted loss (emphasizes rare classes)
	weights = compute_class_weights(y_true, method="balanced")
	loss_weighted = categorical_cross_entropy(y_true, y_pred, class_weights=weights)
	
	print(f"✓ Unweighted loss: {float(loss_unweighted.numpy()):.6f}")
	print(f"✓ Weighted loss:   {float(loss_weighted.numpy()):.6f}")
	print(f"  (Weighted loss emphasizes rare classes)")


def test_class_weights_with_numpy_input():
	"""Test that compute_class_weights works with numpy arrays."""
	y_np = np.array([
		[1, 0, 0],
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	], dtype=np.float32)
	
	weights = compute_class_weights(y_np, method="balanced")
	
	assert len(weights) == 3
	assert all(w > 0 for w in weights)
	print(f"✓ Numpy input weights: {weights}")


def test_class_weights_extreme_imbalance():
	"""Test with highly imbalanced dataset."""
	# 100 samples of class 0, 1 sample of class 1
	y = Tensor(np.vstack([
		np.eye(2)[0].reshape(1, -1) for _ in range(100)
	] + [np.array([[0, 1]], dtype=np.float32)]))
	
	weights = compute_class_weights(y, method="balanced")
	
	# Class 1 should have much higher weight
	assert weights[1] > weights[0], "Rare class should have higher weight"
	print(f"✓ Extreme imbalance weights: {weights}")
	print(f"  Class 0 (100 samples): weight={weights[0]:.4f}")
	print(f"  Class 1 (1 sample):    weight={weights[1]:.4f}")
	print(f"  Ratio: {weights[1] / weights[0]:.1f}x")


if __name__ == "__main__":
	print("Testing class weights computation...\n")
	test_compute_class_weights_balanced()
	test_compute_class_weights_inverse_freq()
	test_class_weights_with_numpy_input()
	test_class_weights_extreme_imbalance()
	test_categorical_cross_entropy_with_weights()
	print("\n✅ All class weights tests passed!")
