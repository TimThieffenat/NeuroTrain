"""Tensor class with minimal autograd support for educational projects."""

from typing import Any, Iterable

import numpy as np


def _unbroadcast_grad(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted gradient back to the original operand shape."""

    if grad.shape == shape:
        return grad

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Tensor:
    """Small wrapper around ``numpy.ndarray`` with reverse-mode autodiff."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        dtype: Any | None = None,
        copy: bool = True,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        source = data.data if isinstance(data, Tensor) else data
        if copy:
            array = np.array(source, dtype=dtype, copy=True)
        else:
            array = np.asarray(source, dtype=dtype)

        self.data = array
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}, data={self.data!r})"
        )

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        if dtype is None:
            return self.data
        return np.asarray(self.data, dtype=dtype)

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def copy(self) -> "Tensor":
        copied = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        if self.grad is not None:
            copied.grad = np.array(self.grad, copy=True)
        return copied

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Any | None = None) -> None:
        """Compute gradients from this tensor down to graph leaves."""

        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar tensors")
            grad_array = np.ones_like(self.data)
        else:
            grad_array = np.asarray(grad)
            if grad_array.shape != self.data.shape:
                raise ValueError("grad shape must match tensor shape")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = grad_array

        for node in reversed(topo):
            node._backward()

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

    @staticmethod
    def _to_tensor(value: Any) -> "Tensor":
        if isinstance(value, Tensor):
            return value
        return Tensor(value)

    @staticmethod
    def _to_array(value: Any) -> np.ndarray:
        if isinstance(value, Tensor):
            return value.data
        return np.asarray(value)

    def astype(self, dtype: Any) -> "Tensor":
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        axes_used = axes if axes else tuple(range(self.ndim - 1, -1, -1))
        out = Tensor(
            self.data.transpose(*axes_used),
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            inverse_axes = np.argsort(axes_used)
            self._accumulate_grad(np.transpose(out.grad, axes=inverse_axes))

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                grad_self = np.broadcast_to(grad, self.shape)
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(ax if ax >= 0 else ax + self.ndim for ax in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad_self = np.broadcast_to(grad, self.shape)

            self._accumulate_grad(grad_self)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            divisor = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            axes = tuple(ax if ax >= 0 else ax + self.ndim for ax in axes)
            divisor = int(np.prod([self.shape[ax] for ax in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def __getitem__(self, item: Any) -> "Tensor":
        out = Tensor(
            self.data[item],
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="slice",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = np.zeros_like(self.data)
            np.add.at(grad_self, item, out.grad)
            self._accumulate_grad(grad_self)

        out._backward = _backward
        return out

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = self._to_array(value)

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(-out.grad)

        out._backward = _backward
        return out

    def __add__(self, other: Any) -> "Tensor":
        other_tensor = self._to_tensor(other)
        out = Tensor(
            self.data + other_tensor.data,
            copy=False,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _children=(self, other_tensor),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast_grad(out.grad, self.shape))
            if other_tensor.requires_grad:
                other_tensor._accumulate_grad(_unbroadcast_grad(out.grad, other_tensor.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-self._to_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return self._to_tensor(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        other_tensor = self._to_tensor(other)
        out = Tensor(
            self.data * other_tensor.data,
            copy=False,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _children=(self, other_tensor),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self_grad = out.grad * other_tensor.data
                self._accumulate_grad(_unbroadcast_grad(self_grad, self.shape))
            if other_tensor.requires_grad:
                other_grad = out.grad * self.data
                other_tensor._accumulate_grad(_unbroadcast_grad(other_grad, other_tensor.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        other_tensor = self._to_tensor(other)
        out = Tensor(
            self.data / other_tensor.data,
            copy=False,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _children=(self, other_tensor),
            _op="div",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self_grad = out.grad / other_tensor.data
                self._accumulate_grad(_unbroadcast_grad(self_grad, self.shape))
            if other_tensor.requires_grad:
                other_grad = -out.grad * self.data / (other_tensor.data ** 2)
                other_tensor._accumulate_grad(_unbroadcast_grad(other_grad, other_tensor.shape))

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        return self._to_tensor(other) / self

    def __matmul__(self, other: Any) -> "Tensor":
        other_tensor = self._to_tensor(other)
        out = Tensor(
            self.data @ other_tensor.data,
            copy=False,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _children=(self, other_tensor),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self_grad = out.grad @ np.swapaxes(other_tensor.data, -1, -2)
                self._accumulate_grad(_unbroadcast_grad(self_grad, self.shape))
            if other_tensor.requires_grad:
                other_grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other_tensor._accumulate_grad(_unbroadcast_grad(other_grad, other_tensor.shape))

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Any) -> "Tensor":
        return self._to_tensor(other) @ self

    def exp(self) -> "Tensor":
        out = Tensor(
            np.exp(self.data),
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            copy=False,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), copy=False, requires_grad=self.requires_grad, _children=(self,), _op="abs")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * np.sign(self.data))

        out._backward = _backward
        return out

    def clip(self, min_value: float | None = None, max_value: float | None = None) -> "Tensor":
        clipped = np.clip(self.data, min_value, max_value)
        out = Tensor(clipped, copy=False, requires_grad=self.requires_grad, _children=(self,), _op="clip")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            mask = np.ones_like(self.data, dtype=float)
            if min_value is not None:
                mask = mask * (self.data >= min_value)
            if max_value is not None:
                mask = mask * (self.data <= max_value)
            self._accumulate_grad(out.grad * mask)

        out._backward = _backward
        return out

    @classmethod
    def zeros(cls, shape: int | Iterable[int], dtype: Any = float, requires_grad: bool = False) -> "Tensor":
        return cls(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape: int | Iterable[int], dtype: Any = float, requires_grad: bool = False) -> "Tensor":
        return cls(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def arange(cls, *args: Any, requires_grad: bool = False, **kwargs: Any) -> "Tensor":
        return cls(np.arange(*args, **kwargs), requires_grad=requires_grad)
