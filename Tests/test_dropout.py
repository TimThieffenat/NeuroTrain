"""Test dropout layer functionality."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from Core.dropout import Dropout
from Core.tensor import Tensor


def test_dropout_shape():
	"""Test that dropout preserves tensor shape."""
	dropout = Dropout(p=0.5)
	x = Tensor(np.random.randn(32, 128))
	y = dropout(x)
	
	assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
	print("✓ Dropout preserves shape")


def test_dropout_training_vs_eval():
	"""Test that dropout behaves differently in training vs eval mode."""
	dropout = Dropout(p=0.5)
	x = Tensor(np.ones((1000, 1000)))  # Large tensor for averaging
	
	# Training mode: should see ~50% zeros (from dropout)
	dropout.train()
	y_train = dropout(x)
	zero_fraction_train = np.mean(y_train.data == 0)
	
	# Eval mode: should see no zeros (no dropout)
	dropout.eval()
	y_eval = dropout(x)
	zero_fraction_eval = np.mean(y_eval.data == 0)
	
	assert 0.45 < zero_fraction_train < 0.55, f"Training dropout rate too far from 50%: {zero_fraction_train:.2%}"
	assert zero_fraction_eval == 0, f"Eval mode should have no zeros, got {zero_fraction_eval:.2%}"
	print(f"✓ Training dropout rate: {zero_fraction_train:.2%} (expected ~50%)")
	print(f"✓ Eval mode dropout rate: {zero_fraction_eval:.2%} (expected 0%)")


def test_dropout_p_zero():
	"""Test that p=0 means no dropout."""
	dropout = Dropout(p=0.0)
	dropout.train()  # Even in training mode
	x = Tensor(np.ones((100, 100)))
	y = dropout(x)
	
	assert np.allclose(y.data, x.data), "p=0 should not drop any values"
	print("✓ p=0 disables dropout completely")


def test_dropout_invalid_p():
	"""Test that invalid p raises error."""
	try:
		Dropout(p=1.0)
		assert False, "Should raise ValueError for p=1.0"
	except ValueError:
		print("✓ p=1.0 correctly raises ValueError")
	
	try:
		Dropout(p=-0.1)
		assert False, "Should raise ValueError for p<0"
	except ValueError:
		print("✓ p<0 correctly raises ValueError")


def test_dropout_scaling():
	"""Test that dropout applies inverted dropout scaling (1/(1-p))."""
	dropout = Dropout(p=0.5)
	dropout.train()
	
	x = Tensor(np.ones((10000, 100)))
	y = dropout(x)
	
	# With inverted dropout, non-zero values should be scaled by 2 (1/(1-0.5))
	non_zero_values = y.data[y.data != 0]
	expected_scale = 1.0 / (1.0 - 0.5)
	mean_value = np.mean(non_zero_values)
	
	assert np.isclose(mean_value, expected_scale, rtol=0.05), \
		f"Expected mean non-zero value ~{expected_scale}, got {mean_value}"
	print(f"✓ Inverted dropout scaling correct: non-zero mean = {mean_value:.2f} (expected {expected_scale})")


if __name__ == "__main__":
	print("Testing Dropout layer...\n")
	test_dropout_shape()
	test_dropout_training_vs_eval()
	test_dropout_p_zero()
	test_dropout_invalid_p()
	test_dropout_scaling()
	print("\n✅ All dropout tests passed!")
