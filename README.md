# NeuroTrain

NeuroTrain is a minimal framework to learn deep learning internals step by step. It's purposly made without any external machine learning library.


## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Run Tests

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

- Tensor operations
- Basi AutoGrad
- Linear layers
- Activations
- Common losses
- Optimizers
- Dataset
- Training loop
- Exportation
- DropOut
- Normalization
- Class Weigth

Not implemented yet:

- CNN
- LSTM
- GRU
- XLSTM
- MaxPool
- Flatten
- Regularization
- K-Split Validation
- Data Agmentation
- PostProcess
