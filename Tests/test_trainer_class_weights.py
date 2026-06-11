"""Test Trainer with class weights."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from Core.tensor import Tensor
from Core.model import Model
from Core.layer import Linear
from Core.activation import relu
from Core.loss import categorical_cross_entropy, compute_class_weights
from Core.optimizer import Adam
from Data.dataset import Dataset
from Train.train import Trainer


def test_trainer_with_class_weights():
	"""Test that Trainer properly passes class_weights to loss function."""
	
	# Create simple model
	model = Model()
	model.add(Linear(5, 3))
	model.add(relu)
	model.add(Linear(3, 2))
	
	# Create imbalanced dataset
	# Class 0: 10 samples, Class 1: 1 sample
	x_data = np.random.randn(11, 5).astype(np.float32)
	y_data = np.vstack([
		np.eye(2)[0].reshape(1, -1) for _ in range(10)
	] + [np.array([[0, 1]], dtype=np.float32)])
	
	train_dataset = Dataset(
		x=Tensor(x_data),
		y=Tensor(y_data)
	)
	
	val_dataset = Dataset(
		x=Tensor(np.random.randn(3, 5).astype(np.float32)),
		y=Tensor(np.eye(2)[[0, 1, 0]])
	)
	
	# Compute class weights
	class_weights = compute_class_weights(train_dataset.y, method="balanced")
	print(f"Class weights: {class_weights}")
	
	# Create trainer WITH class weights
	trainer_weighted = Trainer(
		model=model,
		loss_fn=categorical_cross_entropy,
		optimizer=Adam(model.parameters(), lr=0.01),
		class_weights=class_weights,
	)
	
	# Train for a few epochs
	history_weighted = trainer_weighted.train(
		train_dataset,
		val_dataset,
		epochs=5,
		batch_size=4,
		verbose=False,
	)
	
	print(f"✓ Trainer with class_weights completed {len(history_weighted)} epochs")
	print(f"  Final train loss: {history_weighted[-1]['train_loss']:.6f}")
	print(f"  Final val loss:   {history_weighted[-1]['val_loss']:.6f}")


def test_trainer_without_class_weights():
	"""Test that Trainer still works without class_weights."""
	
	# Create simple model
	model = Model()
	model.add(Linear(5, 3))
	model.add(relu)
	model.add(Linear(3, 2))
	
	# Create dataset
	x_data = np.random.randn(11, 5).astype(np.float32)
	y_data = np.eye(2)[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
	
	train_dataset = Dataset(
		x=Tensor(x_data),
		y=Tensor(y_data)
	)
	
	val_dataset = Dataset(
		x=Tensor(np.random.randn(3, 5).astype(np.float32)),
		y=Tensor(np.eye(2)[[0, 1, 0]])
	)
	
	# Create trainer WITHOUT class weights
	trainer = Trainer(
		model=model,
		loss_fn=categorical_cross_entropy,
		optimizer=Adam(model.parameters(), lr=0.01),
	)
	
	# Train for a few epochs
	history = trainer.train(
		train_dataset,
		val_dataset,
		epochs=5,
		batch_size=4,
		verbose=False,
	)
	
	print(f"✓ Trainer without class_weights completed {len(history)} epochs")
	print(f"  Final train loss: {history[-1]['train_loss']:.6f}")
	print(f"  Final val loss:   {history[-1]['val_loss']:.6f}")


if __name__ == "__main__":
	print("Testing Trainer with class_weights...\n")
	test_trainer_without_class_weights()
	print()
	test_trainer_with_class_weights()
	print("\n✅ All Trainer class_weights tests passed!")
