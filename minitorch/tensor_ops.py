from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type
import ctypes
import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    IndexingError,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)
        self.tanh_map = ops.map(operators.tanh)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)
        self.pow_scalar_zip = ops.zip(operators.pow)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Matrix multiplication of two tensors.
        
        Args:
            a: First tensor with shape (..., m, n)
            b: Second tensor with shape (..., n, p)
            
        Returns:
            Tensor with shape (..., m, p)
        """
        # Get the shapes
        a_shape = a.shape
        b_shape = b.shape
        
        # Check if we can perform matrix multiplication
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("Both tensors must be at least 2D for matrix multiplication")
        
        # For matrix multiplication, the last two dimensions must be compatible
        # a: (..., m, n), b: (..., n, p) -> result: (..., m, p)
        m, n = a_shape[-2], a_shape[-1]
        n2, p = b_shape[-2], b_shape[-1]
        
        if n != n2:
            raise ValueError(f"Matrix multiplication requires compatible dimensions: {a_shape} and {b_shape}")
        
        # Handle broadcasting for higher dimensions
        # Broadcast the leading dimensions
        if len(a_shape) > 2 or len(b_shape) > 2:
            # Get the leading dimensions (excluding the last 2)
            a_leading = a_shape[:-2]
            b_leading = b_shape[:-2]
            
            # Broadcast the leading dimensions
            try:
                leading_shape = shape_broadcast(a_leading, b_leading)
            except IndexingError:
                raise ValueError(f"Cannot broadcast leading dimensions: {a_leading} and {b_leading}")
            
            # Create output shape
            out_shape = leading_shape + (m, p)
        else:
            out_shape = (m, p)
        
        # Create output tensor
        out = a.zeros(out_shape)
        
        # Get the tensor data
        a_storage, a_shape_array, a_strides = a.tuple()
        b_storage, b_shape_array, b_strides = b.tuple()
        out_storage, out_shape_array, out_strides = out.tuple()
        
        # Perform matrix multiplication
        # For each output position (i, j), compute sum over k
        for i in range(m):
            for j in range(p):
                # Compute C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))
                sum_val = 0.0
                for k in range(n):
                    # Get A[i,k]
                    a_index = np.array([i, k])
                    a_pos = index_to_position(a_index, a_strides)
                    a_val = a_storage[a_pos]
                    
                    # Get B[k,j]
                    b_index = np.array([k, j])
                    b_pos = index_to_position(b_index, b_strides)
                    b_val = b_storage[b_pos]
                    
                    # Accumulate the product
                    sum_val += a_val * b_val
                
                # Set the result
                out_index = np.array([i, j])
                out_pos = index_to_position(out_index, out_strides)
                out_storage[out_pos] = sum_val
        
        return out

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        in_index: Index = np.zeros(MAX_DIMS, np.int16)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        a_index: Index = np.zeros(MAX_DIMS, np.int32)
        b_index: Index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float

    Returns:
        Tensor reduce function.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return _reduce


# Load the shared library
try:
    lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
except:
    print("cuda kernels not implemented: combine.so not found")

datatype = np.float32

# function map
fn_map = {
    operators.add: 1,
    operators.mul: 2,
    operators.id: 3,
    operators.neg: 4,
    operators.lt: 5,
    operators.eq: 6,
    operators.sigmoid: 7,
    operators.relu: 8,
    operators.relu_back: 9,
    operators.log: 10,
    operators.log_back: 11,
    operators.exp: 12,
    operators.inv: 13,
    operators.inv_back: 14,
    operators.is_close: 15,
    operators.max: 16,
    operators.pow: 17,
    operators.tanh: 18
}

THREADS_PER_BLOCK = 32


class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Define the argument type for the tensorMap function
            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_strides
                ctypes.c_int,  # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # in_strides
                ctypes.c_int,  # in_size
                ctypes.c_int,  # shape_len
                ctypes.c_int,  # fn_id
            ]

            # Define the return type for the tensorMap function
            lib.tensorMap.restype = None

            # Call the function
            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            # Define the argument type for the tensorZip function
            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_strides
                ctypes.c_int,  # out_size
                ctypes.c_int,  # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # a_strides
                ctypes.c_int,  # a_size
                ctypes.c_int,  # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # b_strides
                ctypes.c_int,  # b_size
                ctypes.c_int,  # b_shape_size
                ctypes.c_int,  # fn_id
            ]

            # Define the return type for the tensorZip function
            lib.tensorZip.restype = None

            # BEGIN ASSIGN2_2
            # TODO
            # 1. Call the tensorZip function implemented in CUDA
            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            # raise NotImplementedError("Zip Function Not Implemented Yet")
            # END ASSIGN2_2

            return out

        return ret

    @staticmethod
    def reduce(
            fn: Callable[[float, float], float], reduce_value: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            # Define the return type for the tensorReduce function
            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_strides
                ctypes.c_int,  # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # in_strides
                ctypes.c_int,  # reduce_dim
                ctypes.c_double,  # reduce_value
                ctypes.c_int,  # shape_len
                ctypes.c_int,  # fn_id
            ]

            # Define the return type for the tensorReduce function
            lib.tensorReduce.restype = None

            # BEGIN ASSIGN2_3
            # TODO
            # 1. Call the tensorReduce function implemented in CUDA
            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                reduce_value,
                len(out.shape),
                fn_id
            )
            # raise NotImplementedError("Reduce Function Not Implemented Yet")
            # END ASSIGN2_3

            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])

        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # b_strides
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # out_shape[1], m
            ctypes.c_int  # out_shape[2], p
        ]

        # Define the return type for the tensorZip function
        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        # BEGIN ASSIGN2_4
        # TODO
        # 1. Call the Matmul function implemented in CUDA
        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            out.shape[1],
            out.shape[2]
        )
        # raise NotImplementedError("Matrix Multiply Function Not Implemented Yet")
        # END ASSIGN2_4

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
        return out


SimpleBackend = TensorBackend(SimpleOps)