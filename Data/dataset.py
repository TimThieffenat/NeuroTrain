"""Dataset utilities for training loops."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np

from Core.tensor import Tensor


@dataclass
class Dataset:
	"""Container for input features ``x`` and targets ``y``.
	
	Attributes:
		x: Input features tensor
		y: Target values tensor
		class_names: Optional mapping of class indices to names (e.g., [0: 'cat', 1: 'dog'])
	"""

	x: Tensor
	y: Tensor
	class_names: list[str] | None = None

	def __post_init__(self) -> None:
		if len(self.x) != len(self.y):
			raise ValueError("x and y must have the same number of samples")

	def __len__(self) -> int:
		return len(self.x)

	def __getitem__(self, index: int | slice) -> tuple[Tensor, Tensor]:
		return self.x[index], self.y[index]

	def batches(self, batch_size: int, shuffle: bool = True) -> Iterator[tuple[Tensor, Tensor]]:
		"""Yield mini-batches as ``(x_batch, y_batch)``."""

		if batch_size <= 0:
			raise ValueError("batch_size must be > 0")

		indices = np.arange(len(self))
		if shuffle:
			np.random.shuffle(indices)

		for start in range(0, len(indices), batch_size):
			batch_indices = indices[start : start + batch_size]
			yield self.x[batch_indices], self.y[batch_indices]


def load_csv(
	file_path: str | Path,
	target_columns: int | list[int] | tuple[int, ...] | None = -1,
	delimiter: str = ",",
	skip_header: bool = False,
	dtype: type | np.dtype = np.float64,
	id_columns: int | list[int] | tuple[int, ...] | None = None,
) -> Dataset:
	"""Load a CSV file and return a ``Dataset``.

	By default, the last column is treated as target and all others as features.
	ID columns (if specified) are dropped before any processing.
	
	If target_columns=None, all columns (except id_columns) are treated as features,
	and a dummy target column of zeros is created.
	"""

	path = Path(file_path)
	if not path.exists():
		raise FileNotFoundError(f"CSV file not found: {path}")

	data = np.genfromtxt(
		path,
		delimiter=delimiter,
		dtype=str,
		skip_header=1 if skip_header else 0,
	)

	if data.ndim == 1:
		data = data.reshape(1, -1)

	if data.shape[1] < 2:
		raise ValueError("CSV must contain at least 2 columns")

	n_cols_original = data.shape[1]

	if id_columns is not None:
		if isinstance(id_columns, int):
			id_indices = [id_columns]
		else:
			id_indices = list(id_columns)

		normalized_ids: list[int] = []
		for idx in id_indices:
			normalized = idx if idx >= 0 else n_cols_original + idx
			if normalized < 0 or normalized >= n_cols_original:
				raise IndexError("id column index out of range")
			normalized_ids.append(normalized)

		if len(set(normalized_ids)) != len(normalized_ids):
			raise ValueError("id_columns contains duplicates")

		keep_indices = [i for i in range(n_cols_original) if i not in normalized_ids]
		data = data[:, keep_indices]

		def remap_indices(orig_indices: list[int], drop_set: set[int], orig_cols: int) -> list[int]:
			"""Normalize indices to original columns, then remap to new columns after dropping."""
			normalized = []
			for idx in orig_indices:
				norm = idx if idx >= 0 else orig_cols + idx
				if norm < 0 or norm >= orig_cols:
					raise IndexError("target column index out of range")
				normalized.append(norm)
			
			remapped = []
			for idx in normalized:
				count_before = sum(1 for d in drop_set if d < idx)
				remapped.append(idx - count_before)
			return remapped

		# Handle target_columns=None case when id_columns exist
		if target_columns is not None:
			if isinstance(target_columns, int):
				target_indices = remap_indices([target_columns], set(normalized_ids), n_cols_original)
			else:
				target_indices = remap_indices(list(target_columns), set(normalized_ids), n_cols_original)
		else:
			target_indices = None
	else:
		if target_columns is not None:
			if isinstance(target_columns, int):
				target_indices = [target_columns]
			else:
				target_indices = list(target_columns)
		else:
			target_indices = None

	# Handle case where target_columns=None (inference mode)
	if target_indices is None:
		# All columns are features, create dummy target
		output_dtype = np.dtype(dtype)
		feature_blocks: list[np.ndarray] = []
		for col in range(data.shape[1]):
			column = data[:, col]
			try:
				numeric_column = column.astype(output_dtype).reshape(-1, 1)
				feature_blocks.append(numeric_column)
			except ValueError:
				if not np.issubdtype(output_dtype, np.floating):
					raise ValueError(
						"String feature columns require a floating dtype for one-hot encoding"
					)
				unique_values, encoded = np.unique(column, return_inverse=True)
				one_hot = np.zeros((data.shape[0], len(unique_values)), dtype=output_dtype)
				one_hot[np.arange(data.shape[0]), encoded] = 1.0
				feature_blocks.append(one_hot)

		x_np = np.concatenate(feature_blocks, axis=1)
		y_np = np.zeros((data.shape[0], 1), dtype=output_dtype)  # Dummy target
		x = Tensor(x_np)
		y = Tensor(y_np)
		return Dataset(x, y, class_names=None)
	
	if not target_indices:
		raise ValueError("target_columns cannot be empty")

	normalized_targets: list[int] = []
	n_cols = data.shape[1]
	
	if id_columns is None:
		# Only normalize if we haven't already done so above
		for idx in target_indices:
			normalized = idx if idx >= 0 else n_cols + idx
			if normalized < 0 or normalized >= n_cols:
				raise IndexError("target column index out of range")
			normalized_targets.append(normalized)
	else:
		# Already normalized and remapped above
		normalized_targets = target_indices

	if len(set(normalized_targets)) != len(normalized_targets):
		raise ValueError("target_columns contains duplicates")

	target_set = set(normalized_targets)
	feature_indices = [i for i in range(n_cols) if i not in target_set]
	if not feature_indices:
		raise ValueError("No feature columns left after selecting targets")

	feature_data = data[:, feature_indices]
	target_data = data[:, normalized_targets]

	if feature_data.ndim == 1:
		feature_data = feature_data.reshape(-1, 1)

	output_dtype = np.dtype(dtype)
	allow_one_hot = np.issubdtype(output_dtype, np.floating)

	feature_blocks: list[np.ndarray] = []
	for col in range(feature_data.shape[1]):
		column = feature_data[:, col]
		try:
			numeric_column = column.astype(output_dtype).reshape(-1, 1)
			feature_blocks.append(numeric_column)
		except ValueError:
			if not allow_one_hot:
				raise ValueError(
					"String feature columns require a floating dtype for one-hot encoding"
				)
			unique_values, encoded = np.unique(column, return_inverse=True)
			one_hot = np.zeros((feature_data.shape[0], len(unique_values)), dtype=output_dtype)
			one_hot[np.arange(feature_data.shape[0]), encoded] = 1.0
			feature_blocks.append(one_hot)

	x_np = np.concatenate(feature_blocks, axis=1)

	if target_data.ndim == 1:
		target_data = target_data.reshape(-1, 1)

	target_blocks: list[np.ndarray] = []
	class_names_list: list[str] | None = None
	
	for col in range(target_data.shape[1]):
		column = target_data[:, col]
		try:
			numeric_column = column.astype(output_dtype).reshape(-1, 1)
			target_blocks.append(numeric_column)
		except ValueError:
			if not allow_one_hot:
				raise ValueError(
					"String target columns require a floating dtype for one-hot encoding"
				)
			unique_values, encoded = np.unique(column, return_inverse=True)
			one_hot = np.zeros((target_data.shape[0], len(unique_values)), dtype=output_dtype)
			one_hot[np.arange(target_data.shape[0]), encoded] = 1.0
			target_blocks.append(one_hot)
			
			# Save class names from first target column
			if class_names_list is None:
				class_names_list = sorted(list(unique_values))

	y_np = np.concatenate(target_blocks, axis=1)

	x = Tensor(x_np)
	y = Tensor(y_np)
	return Dataset(x, y, class_names=class_names_list)


def train_val_split(
	dataset: Dataset,
	val_ratio: float = 0.2,
	shuffle: bool = True,
) -> tuple[Dataset, Dataset]:
	"""Split a dataset into train and validation datasets."""

	if not isinstance(dataset, Dataset):
		raise TypeError("dataset must be a Dataset instance")
	if not 0.0 < val_ratio < 1.0:
		raise ValueError("val_ratio must be between 0 and 1")

	indices = np.arange(len(dataset))
	if shuffle:
		np.random.shuffle(indices)

	val_size = int(len(indices) * val_ratio)
	val_indices = indices[:val_size]
	train_indices = indices[val_size:]

	train_dataset = Dataset(dataset.x[train_indices], dataset.y[train_indices], class_names=dataset.class_names)
	val_dataset = Dataset(dataset.x[val_indices], dataset.y[val_indices], class_names=dataset.class_names)
	return train_dataset, val_dataset
