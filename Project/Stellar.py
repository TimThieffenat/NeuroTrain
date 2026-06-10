"""
Stellar Classification Model Training and Inference Pipeline

This script trains a neural network to classify stellar objects (GALAXY, STAR, QSO)
using the StellarClassPrediction dataset. It supports three modes:
  - training: Train and save a model
  - evaluation: Evaluate the model on validation data
  - inference: Make predictions on unlabeled test data and save results

The model uses:
  - LayerNorm for input normalization
  - Two hidden layers with ReLU activation
  - Softmax output for multi-class classification
  - Adam optimizer with categorical cross-entropy loss
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.normalization import LayerNorm
from Core.activation import relu, softmax
from Core.layer import Linear
from Core.loss import categorical_cross_entropy
from Core.model import Model
from Core.optimizer import Adam
from Data.dataset import load_csv, train_val_split
from Evaluation.eval import Tester
from Export.model_io import load_model, save_model, save_predictions
from Train.train import Trainer


# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
data_path = "C:/Tim/NeuroT/Project/StellarClassPrediction/database/train.csv"
test_data_path = "C:/Tim/NeuroT/Project/StellarClassPrediction/database/test.csv"
model_path = "C:/Tim/NeuroT/Project/StellarClassPrediction/Models/stellar_mlp.json"
result_path = "C:/Tim/NeuroT/Project/StellarClassPrediction/Results/predictions.csv"
loss_plot_path = "C:/Tim/NeuroT/Project/StellarClassPrediction/Results/loss_history.png"
accuracy_plot_path = "C:/Tim/NeuroT/Project/StellarClassPrediction/Results/accuracy_history.png"

# Execution flags
TRAIN = False      # Train and save a new model
EVAL = True       # Evaluate the trained model on validation data
INFERENCE = True  # Make predictions on test data

# Early stopping settings (monitor validation loss)
EARLY_STOPPING = False
PATIENCE = 5
MIN_DELTA = 1e-4


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

# Load training data with ID column dropped (prevents feature saturation)
dataset = load_csv(
    data_path,
    target_columns=-1,  # Last column is the target (class label)
    delimiter=",",
    skip_header=True,
    id_columns=0,  # Drop the ID column (column 0)
)

# Split into training (80%) and validation (20%) datasets
# Preserves class name mapping from original dataset
train_dataset, val_dataset = train_val_split(dataset, val_ratio=0.2, shuffle=True)


# ============================================================================
# TRAINING
# ============================================================================

if TRAIN:
    # Get dimensions from training data
    input_dim = train_dataset.x.shape[1]
    output_dim = train_dataset.y.shape[1]

    # Build model architecture
    model = Model()
    model.add(LayerNorm(input_dim))           # Normalize input features
    model.add(Linear(input_dim, 32))          # Hidden layer: 32 neurons
    model.add(relu)                           # ReLU activation
    model.add(Linear(32, 32))                 # Hidden layer: 32 neurons
    model.add(relu)                           # ReLU activation
    model.add(Linear(32, 32))                 # Hidden layer: 32 neurons
    model.add(relu)                           # ReLU activation
    model.add(Linear(32, output_dim))         # Output layer: one neuron per class
    model.add(softmax)                        # Softmax for multi-class probabilities

    # Configure trainer and optimizer
    trainer = Trainer(
        model=model,
        loss_fn=categorical_cross_entropy,
        optimizer=Adam(model.parameters(), lr=0.001),
    )

    # Train with validation-based early stopping and best-weight restore
    trainer.train(
        train_dataset,
        val_dataset,
        epochs=200,
        batch_size=128,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
        save_metrics=["train_loss", "val_loss", "train_accuracy", "val_accuracy"],
    )

    trainer.plot_history(
        metric_name="loss",
        title="Training and Validation Loss",
        save_path=loss_plot_path)
    
    trainer.plot_history(
        metric_name="accuracy",
        title="Training and Validation Accuracy",
        save_path=accuracy_plot_path,
    )

    # Save trained model for later use
    save_model(model, model_path)


# ============================================================================
# EVALUATION
# ============================================================================

if EVAL:
    # Load previously trained model
    model = load_model(model_path)

    # Initialize tester for evaluation
    tester = Tester(model=model, loss_fn=categorical_cross_entropy)

    # Evaluate on validation dataset
    metrics = tester.test(
        val_dataset,
        metrics=["accuracy", "precision", "recall", "f1", "balanced_accuracy"],
        batch_size=128,
        average="macro",  # Macro-average across all classes
    )

    # Print evaluation results
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


# ============================================================================
# INFERENCE
# ============================================================================

if INFERENCE:
    # Load trained model
    model = load_model(model_path)

    # Load test data without labels (target_columns=None)
    # Drop ID column to match training preprocessing
    test_dataset = load_csv(
        test_data_path,
        target_columns=None,  # No labels in test data
        delimiter=",",
        skip_header=True,
        id_columns=0,  # Drop ID column
    )

    # Generate predictions on test data
    y_pred = model.predict(test_dataset.x)

    # Save predictions as CSV with original IDs and decoded class names
    # Uses class_names mapping from training dataset for correct decoding
    save_predictions(
        test_data_path,
        y_pred,
        result_path,
        class_names=train_dataset.class_names,  # Use class mapping from training
        id_column=0,
        delimiter=",",
        skip_header=True,
    )

