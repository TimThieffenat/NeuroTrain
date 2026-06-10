# NeuroTrain

NeuroTrain is a minimal framework to learn deep learning internals step by step.
The goal is educational clarity: start from tensors, then layers, losses,
optimizers, datasets, and autograd.

## Goals

- Understand how a tensor object is built.
- Build a simple MLP from basic layers.
- Implement losses and optimizers with explicit code.
- Train with a clear loop using train/validation datasets.
- Learn reverse-mode autograd without hiding complexity.

## Project Structure

```
Core/
	tensor.py        # Tensor + autograd engine
	layer.py         # Layer abstractions + Linear
	activation.py    # relu, sigmoid, tanh
	loss.py          # MSE, MAE, BCE, CCE
	optimizer.py     # SGD, Adam
Data/
	dataset.py       # Dataset container + train/val split
Evaluations/
	eval.py          # Evaluation helpers (metrics on datasets)
Project/
	mlp.py           # Minimal MLP model
Train/
	train.py         # Trainer loop
```

## Requirements

- Python 3.10+
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Run Tests

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run the full unit test suite from the project root:

```bash
python -m pytest -q
```

Run a single test file:

```bash
python -m pytest tests/test_optimizers.py -q
```

## Quick Start

Example: tiny regression training with MLP + autograd.

```python
import numpy as np

from Core.loss import mse_loss
from Core.optimizer import SGD
from Core.tensor import Tensor
from Data.dataset import Dataset, train_val_split
from Evaluation.eval import eval
from Project.mlp import MLP
from Train.train import Trainer

# Fake data: y = 2*x0 - x1
x_np = np.random.randn(200, 2)
y_np = (2.0 * x_np[:, :1] - x_np[:, 1:2])

x = Tensor(x_np)
y = Tensor(y_np)
dataset = Dataset(x, y)

train_ds, val_ds = train_val_split(dataset, val_ratio=0.2)

model = MLP([2, 8, 1])
optimizer = SGD(model.parameters(), lr=0.01)
trainer = Trainer(model=model, loss_fn=mse_loss, optimizer=optimizer)

history = trainer.train(
		train_dataset=train_ds,
		val_dataset=val_ds,
		epochs=20,
		batch_size=16,
)

print(history[-1])

metrics = eval(model=model, dataset=val_ds, metrics=["mse", "rmse", "mae", "r2"])
print(metrics)
```

## Current Scope

Implemented:

- Tensor operations with basic autograd
- Linear layers and activations
- Common losses and optimizers
- Dataset and training loop

Not implemented yet:

- Advanced layers (Conv, RNN, etc.)
- Advanced regularization and schedulers
- GPU acceleration
- Full test suite and CI pipeline

## Next Steps

- Add more unit tests in `tests/`
- Add examples in a dedicated `examples/` folder
- Improve packaging with `pyproject.toml`
