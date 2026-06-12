import numpy as np

from Core.activation import relu
from Core.layer import Linear
from Core.model import NeuralNetwork
from Core.loss import mse_loss
from Core.optimizer import SGD
from Core.tensor import Tensor
from Data.dataset import Dataset, load_csv, train_val_split
from Train.train import NeuralNetworkTrainer


def build_model() -> NeuralNetwork:
    model = NeuralNetwork()
    model.add(Linear(2, 6))
    model.add(relu)
    model.add(Linear(6, 1))
    return model


def test_train_val_split_preserves_total_size():
    x = Tensor(np.random.randn(50, 2))
    y = Tensor(np.random.randn(50, 1))
    ds = Dataset(x, y)

    train_ds, val_ds = train_val_split(ds, val_ratio=0.2, shuffle=False)

    assert len(train_ds) == 40
    assert len(val_ds) == 10
    assert len(train_ds) + len(val_ds) == 50


def test_dataset_batches_cover_all_samples():
    x = Tensor(np.arange(20).reshape(10, 2))
    y = Tensor(np.arange(10).reshape(10, 1))
    ds = Dataset(x, y)

    seen = 0
    for xb, yb in ds.batches(batch_size=4, shuffle=False):
        assert xb.shape[0] == yb.shape[0]
        seen += xb.shape[0]

    assert seen == len(ds)


def test_dataset_load_csv_uses_last_column_as_target(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,10\n3,4,20\n")

    ds = load_csv(csv_path)

    assert len(ds) == 2
    assert ds.x.shape == (2, 2)
    assert ds.y.shape == (2, 1)
    np.testing.assert_allclose(ds.x.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(ds.y.numpy(), np.array([[10.0], [20.0]]))


def test_dataset_load_csv_supports_custom_target_columns(tmp_path):
    csv_path = tmp_path / "sample_multi_target.csv"
    csv_path.write_text("1,9,2,10\n3,19,4,20\n")

    ds = load_csv(csv_path, target_columns=[1, 3])

    assert ds.x.shape == (2, 2)
    assert ds.y.shape == (2, 2)
    np.testing.assert_allclose(ds.x.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(ds.y.numpy(), np.array([[9.0, 10.0], [19.0, 20.0]]))


def test_dataset_load_csv_encodes_categorical_target(tmp_path):
    csv_path = tmp_path / "sample_categorical_target.csv"
    csv_path.write_text("f1,f2,label\n1.0,2.0,M\n3.0,4.0,F\n5.0,6.0,M\n")

    ds = load_csv(csv_path, target_columns=-1, skip_header=True)

    assert ds.x.shape == (3, 2)
    assert ds.y.shape == (3, 2)
    np.testing.assert_allclose(ds.x.numpy(), np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    # Encodage one-hot attendu avec np.unique: F->index 0, M->index 1
    np.testing.assert_allclose(ds.y.numpy(), np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))


def test_dataset_load_csv_one_hot_encodes_categorical_features(tmp_path):
    csv_path = tmp_path / "sample_categorical_feature.csv"
    csv_path.write_text("color,size,label\nred,1.0,10\nblue,2.0,20\nred,3.0,30\n")

    ds = load_csv(csv_path, target_columns=-1, skip_header=True)

    # Categorical feature "color" is label-encoded (1 column), not one-hot (2 columns)
    assert ds.x.shape == (3, 2)
    assert ds.y.shape == (3, 1)
    # np.unique ordonne les categories: blue=0, red=1
    expected_x = np.array(
        [
            [1.0, 1.0],
            [0.0, 2.0],
            [1.0, 3.0],
        ]
    )
    np.testing.assert_allclose(ds.x.numpy(), expected_x)
    np.testing.assert_allclose(ds.y.numpy(), np.array([[10.0], [20.0], [30.0]]))


def test_dataset_load_csv_drops_id_columns(tmp_path):
    csv_path = tmp_path / "sample_with_id.csv"
    csv_path.write_text("id,feature1,feature2,label\n1,2.0,3.0,10\n2,4.0,5.0,20\n3,6.0,7.0,30\n")

    ds = load_csv(csv_path, target_columns=-1, skip_header=True, id_columns=0)

    assert ds.x.shape == (3, 2)
    assert ds.y.shape == (3, 1)
    np.testing.assert_allclose(ds.x.numpy(), np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    np.testing.assert_allclose(ds.y.numpy(), np.array([[10.0], [20.0], [30.0]]))


def test_dataset_load_csv_drops_multiple_id_columns(tmp_path):
    csv_path = tmp_path / "sample_with_multi_id.csv"
    csv_path.write_text("id1,feature1,id2,feature2,label\n1,2.0,100,3.0,10\n2,4.0,200,5.0,20\n")

    ds = load_csv(csv_path, target_columns=-1, skip_header=True, id_columns=[0, 2])

    assert ds.x.shape == (2, 2)
    assert ds.y.shape == (2, 1)
    np.testing.assert_allclose(ds.x.numpy(), np.array([[2.0, 3.0], [4.0, 5.0]]))
    np.testing.assert_allclose(ds.y.numpy(), np.array([[10.0], [20.0]]))


def test_trainer_runs_and_returns_history():
    np.random.seed(0)

    x_np = np.random.randn(80, 2)
    y_np = (2.0 * x_np[:, :1] - x_np[:, 1:2])

    x = Tensor(x_np)
    y = Tensor(y_np)
    ds = Dataset(x, y)
    train_ds, val_ds = train_val_split(ds, val_ratio=0.25, shuffle=True)

    model = build_model()
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = NeuralNetworkTrainer(model=model, loss_fn=mse_loss, optimizer=optimizer)

    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=3,
        batch_size=16,
        verbose=False,
    )

    assert len(history) == 3
    assert "train_loss" in history[-1]
    assert "val_loss" in history[-1]
    assert np.isfinite(history[-1]["train_loss"])
    assert np.isfinite(history[-1]["val_loss"])


def test_trainer_early_stopping_stops_before_max_epochs():
    np.random.seed(0)

    x_np = np.random.randn(60, 2)
    y_np = (1.5 * x_np[:, :1] - 0.5 * x_np[:, 1:2])

    ds = Dataset(Tensor(x_np), Tensor(y_np))
    train_ds, val_ds = train_val_split(ds, val_ratio=0.2, shuffle=False)

    model = build_model()
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = NeuralNetworkTrainer(model=model, loss_fn=mse_loss, optimizer=optimizer)

    # Force non-improving validation to deterministically trigger early stopping.
    trainer.evaluate_loss = lambda _dataset, batch_size=32: 1.0  # type: ignore[assignment]

    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=10,
        batch_size=16,
        verbose=False,
        early_stopping=True,
        patience=2,
    )

    # First epoch sets best score, then 2 non-improving epochs -> stop at epoch 3.
    assert len(history) == 3


def test_trainer_early_stopping_validates_arguments():
    ds = Dataset(Tensor(np.random.randn(10, 2)), Tensor(np.random.randn(10, 1)))
    train_ds, val_ds = train_val_split(ds, val_ratio=0.2, shuffle=False)

    model = build_model()
    trainer = NeuralNetworkTrainer(model=model, loss_fn=mse_loss, optimizer=SGD(model.parameters(), lr=0.01))

    try:
        trainer.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            epochs=1,
            verbose=False,
            early_stopping=True,
            patience=0,
        )
        assert False, "Expected ValueError for patience=0"
    except ValueError as exc:
        assert "patience" in str(exc)

    try:
        trainer.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            epochs=1,
            verbose=False,
            early_stopping=True,
            min_delta=-1e-6,
        )
        assert False, "Expected ValueError for negative min_delta"
    except ValueError as exc:
        assert "min_delta" in str(exc)


def test_trainer_saves_requested_metrics_in_attribute():
    np.random.seed(1)

    x_np = np.random.randn(40, 2)
    y_np = (x_np[:, :1] > x_np[:, 1:2]).astype(float)

    ds = Dataset(Tensor(x_np), Tensor(y_np))
    train_ds, val_ds = train_val_split(ds, val_ratio=0.2, shuffle=False)

    model = build_model()
    trainer = NeuralNetworkTrainer(model=model, loss_fn=mse_loss, optimizer=SGD(model.parameters(), lr=0.01))

    trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=2,
        batch_size=8,
        verbose=False,
        save_metrics=["train_loss", "val_loss", "train_accuracy", "val_accuracy"],
    )

    assert "train_loss" in trainer.metrics_history
    assert "val_loss" in trainer.metrics_history
    assert "train_accuracy" in trainer.metrics_history
    assert "val_accuracy" in trainer.metrics_history
    assert len(trainer.metrics_history["train_loss"]) == 2
    assert len(trainer.metrics_history["val_accuracy"]) == 2
