"""Morphogen GPU Dialect (v0.12.0 Phase 6)

This module defines the Morphogen GPU dialect for MLIR, providing high-level
GPU operations that lower to MLIR's built-in GPU dialect for parallel execution.

Status: Phase 6 Implementation - GPU Acceleration

Operations:
- morphogen.gpu.launch: Launch GPU kernel with block/thread configuration
- morphogen.gpu.parallel: Express data-parallel computation
- morphogen.gpu.alloc: Allocate GPU global memory
- morphogen.gpu.alloc_shared: Allocate GPU shared memory
- morphogen.gpu.sync: Thread synchronization barrier
- morphogen.gpu.thread_id: Get thread/block index

Type System:
- !morphogen.gpu.kernel<T>: GPU kernel handle
- !morphogen.gpu.buffer<T>: GPU global memory buffer
- !morphogen.gpu.shared<T>: GPU shared memory buffer

Design Principles:
- Express parallelism structurally (not implicitly)
- Model memory hierarchy explicitly (global/shared/registers)
- Follow canonical GPU lowering pipeline
- Support determinism profiles (strict/repro/live)
"""

from __future__ import annotations
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MorphogenMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, arith, memref, scf, gpu
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class GPUKernelType:
    """Wrapper for !morphogen.gpu.kernel<T> type.

    Represents a GPU kernel configuration with element type.

    Example:
        >>> ctx = MorphogenMLIRContext()
        >>> kernel_type = GPUKernelType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(kernel_type)  # !morphogen.gpu.kernel<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get GPU kernel type for given element type.

        Args:
            element_type: MLIR element type (e.g., F32Type, F64Type)
            context: MLIR context

        Returns:
            Opaque kernel type !morphogen.gpu.kernel<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"gpu.kernel<{element_str}>", context=context)


class GPUBufferType:
    """Wrapper for !morphogen.gpu.buffer<T> type.

    Represents GPU global memory buffer.

    Example:
        >>> buffer_type = GPUBufferType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(buffer_type)  # !morphogen.gpu.buffer<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get GPU buffer type for given element type.

        Args:
            element_type: MLIR element type
            context: MLIR context

        Returns:
            Opaque buffer type !morphogen.gpu.buffer<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"gpu.buffer<{element_str}>", context=context)


class GPUSharedType:
    """Wrapper for !morphogen.gpu.shared<T> type.

    Represents GPU shared memory buffer (faster than global, limited size).

    Example:
        >>> shared_type = GPUSharedType.get(ir.F32Type.get(), ctx.ctx)
        >>> print(shared_type)  # !morphogen.gpu.shared<f32>
    """

    @staticmethod
    def get(element_type: Any, context: Any) -> Any:
        """Get GPU shared memory type for given element type.

        Args:
            element_type: MLIR element type
            context: MLIR context

        Returns:
            Opaque shared type !morphogen.gpu.shared<T>
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        element_str = str(element_type)
        return ir.OpaqueType.get("morphogen", f"gpu.shared<{element_str}>", context=context)


class GPULaunchOp:
    """Operation: morphogen.gpu.launch

    Launches a GPU kernel with specified block and thread configuration.

    Syntax:
        %result = morphogen.gpu.launch %blocks_x, %blocks_y, %blocks_z,
                                       %threads_x, %threads_y, %threads_z
                  (%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>)
                  { ... kernel body ... }

    Attributes:
        - blocks: Grid dimensions (x, y, z)
        - threads: Block dimensions (x, y, z)
        - args: Kernel arguments (memrefs, scalars)

    Results:
        - Kernel completion status

    Lowering:
        Lowers to MLIR gpu.launch_func
    """

    @staticmethod
    def create(
        blocks: List[Any],  # [blocks_x, blocks_y, blocks_z]
        threads: List[Any],  # [threads_x, threads_y, threads_z]
        args: List[Any],  # Kernel arguments
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create a GPU launch operation.

        Args:
            blocks: Grid dimensions [x, y, z]
            threads: Block dimensions [x, y, z]
            args: Kernel arguments
            element_type: Element type
            loc: Source location
            ip: Insertion point

        Returns:
            Launch operation result
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        if len(blocks) != 3 or len(threads) != 3:
            raise ValueError("blocks and threads must have 3 dimensions")

        with loc, ip:
            # Create the kernel type
            kernel_type = GPUKernelType.get(element_type, loc.context)

            # Create placeholder operation
            all_operands = blocks + threads + args
            result = builtin.UnrealizedConversionCastOp(
                [kernel_type],
                all_operands
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.gpu.launch", context=loc.context
            )
            result.operation.attributes["blocks"] = ir.ArrayAttr.get([
                ir.IntegerAttr.get(ir.IndexType.get(), 3)
            ])
            result.operation.attributes["threads"] = ir.ArrayAttr.get([
                ir.IntegerAttr.get(ir.IndexType.get(), 3)
            ])

            return result.results[0]


class GPUParallelOp:
    """Operation: morphogen.gpu.parallel

    Expresses a data-parallel computation suitable for GPU execution.

    Syntax:
        %result = morphogen.gpu.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1)
                  in (%input) : memref<?x?xf32> {
            ... computation using %i, %j ...
            morphogen.gpu.yield %value
        } -> memref<?x?xf32>

    This provides structured parallelism that maps cleanly to GPU threads.

    Lowering:
        Lowers to gpu.launch with proper thread/block mapping
    """

    @staticmethod
    def create(
        lower_bounds: List[Any],
        upper_bounds: List[Any],
        inputs: List[Any],
        result_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create a GPU parallel operation.

        Args:
            lower_bounds: Lower bounds for parallel dimensions
            upper_bounds: Upper bounds for parallel dimensions
            inputs: Input operands
            result_type: Result type
            loc: Source location
            ip: Insertion point

        Returns:
            Parallel operation result
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Create placeholder
            all_operands = lower_bounds + upper_bounds + inputs
            result = builtin.UnrealizedConversionCastOp(
                [result_type],
                all_operands
            )

            # Add attributes
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.gpu.parallel", context=loc.context
            )
            result.operation.attributes["rank"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), len(lower_bounds)
            )

            return result.results[0]


class GPUAllocOp:
    """Operation: morphogen.gpu.alloc

    Allocates GPU global memory.

    Syntax:
        %buffer = morphogen.gpu.alloc(%height, %width) : !morphogen.gpu.buffer<f32>

    Lowering:
        Lowers to gpu.alloc or memref.alloc with gpu address space
    """

    @staticmethod
    def create(
        dims: List[Any],
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create a GPU allocation operation.

        Args:
            dims: Dimension sizes
            element_type: Element type
            loc: Source location
            ip: Insertion point

        Returns:
            GPU buffer
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            buffer_type = GPUBufferType.get(element_type, loc.context)

            result = builtin.UnrealizedConversionCastOp(
                [buffer_type],
                dims
            )

            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.gpu.alloc", context=loc.context
            )

            return result.results[0]


class GPUAllocSharedOp:
    """Operation: morphogen.gpu.alloc_shared

    Allocates GPU shared memory (fast, limited size).

    Syntax:
        %shared = morphogen.gpu.alloc_shared(%size) : !morphogen.gpu.shared<f32>

    Shared memory is faster than global but limited (~48KB per block).
    Useful for tiling and data reuse patterns.

    Lowering:
        Lowers to gpu.dynamic_shared_memory or static allocation
    """

    @staticmethod
    def create(
        size: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Create a GPU shared memory allocation.

        Args:
            size: Allocation size
            element_type: Element type
            loc: Source location
            ip: Insertion point

        Returns:
            Shared memory buffer
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            shared_type = GPUSharedType.get(element_type, loc.context)

            result = builtin.UnrealizedConversionCastOp(
                [shared_type],
                [size]
            )

            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.gpu.alloc_shared", context=loc.context
            )

            return result.results[0]


class GPUSyncOp:
    """Operation: morphogen.gpu.sync

    Thread synchronization barrier.

    Syntax:
        morphogen.gpu.sync

    Ensures all threads in a block reach this point before continuing.
    Required when using shared memory to avoid race conditions.

    Lowering:
        Lowers to gpu.barrier
    """

    @staticmethod
    def create(loc: Any, ip: Any) -> None:
        """Create a GPU synchronization barrier.

        Args:
            loc: Source location
            ip: Insertion point
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        with loc, ip:
            # Create a side-effect operation
            result = builtin.UnrealizedConversionCastOp([], [])
            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.gpu.sync", context=loc.context
            )


class GPUThreadIdOp:
    """Operation: morphogen.gpu.thread_id

    Get current thread/block index.

    Syntax:
        %thread_x = morphogen.gpu.thread_id "x"  // Thread ID in X dimension
        %block_y = morphogen.gpu.thread_id "block_y"  // Block ID in Y dimension

    Dimensions: "x", "y", "z" for threads, "block_x", "block_y", "block_z" for blocks

    Lowering:
        Lowers to gpu.thread_id, gpu.block_id
    """

    @staticmethod
    def create(
        dimension: str,  # "x", "y", "z", "block_x", "block_y", "block_z"
        loc: Any,
        ip: Any
    ) -> Any:
        """Create a thread/block ID operation.

        Args:
            dimension: Dimension name
            loc: Source location
            ip: Insertion point

        Returns:
            Index value
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError("MLIR not available")

        valid_dims = {"x", "y", "z", "block_x", "block_y", "block_z"}
        if dimension not in valid_dims:
            raise ValueError(f"Invalid dimension: {dimension}. Must be one of {valid_dims}")

        with loc, ip:
            index_type = ir.IndexType.get()

            result = builtin.UnrealizedConversionCastOp(
                [index_type],
                []
            )

            result.operation.attributes["op_name"] = ir.StringAttr.get(
                "morphogen.gpu.thread_id", context=loc.context
            )
            result.operation.attributes["dimension"] = ir.StringAttr.get(
                dimension, context=loc.context
            )

            return result.results[0]


class GPUDialect:
    """GPU operations dialect.

    This class serves as a namespace for GPU dialect operations
    and provides utility methods for working with GPU types.

    Operations:
        - launch: Launch GPU kernel
        - parallel: Express data-parallel computation
        - alloc: Allocate GPU global memory
        - alloc_shared: Allocate GPU shared memory
        - sync: Thread synchronization barrier
        - thread_id: Get thread/block index

    Example:
        >>> from morphogen.mlir.dialects.gpu_dialect import GPUDialect
        >>>
        >>> # Allocate GPU buffer
        >>> h = arith.ConstantOp(ir.IndexType.get(), 256)
        >>> w = arith.ConstantOp(ir.IndexType.get(), 256)
        >>> buffer = GPUDialect.alloc([h, w], f32, loc, ip)
        >>>
        >>> # Launch parallel kernel
        >>> blocks = [arith.ConstantOp(ir.IndexType.get(), 16) for _ in range(3)]
        >>> threads = [arith.ConstantOp(ir.IndexType.get(), 256) for _ in range(3)]
        >>> kernel = GPUDialect.launch(blocks, threads, [buffer], f32, loc, ip)
    """

    launch = GPULaunchOp.create
    parallel = GPUParallelOp.create
    alloc = GPUAllocOp.create
    alloc_shared = GPUAllocSharedOp.create
    sync = GPUSyncOp.create
    thread_id = GPUThreadIdOp.create

    @staticmethod
    def is_gpu_op(op: Any) -> bool:
        """Check if an operation is a GPU operation.

        Args:
            op: MLIR operation to check

        Returns:
            True if op is a GPU operation
        """
        if not MLIR_AVAILABLE:
            return False

        if not hasattr(op, "attributes"):
            return False

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return False

        op_name = str(op_name_attr)
        return "morphogen.gpu." in op_name

    @staticmethod
    def get_gpu_op_name(op: Any) -> Optional[str]:
        """Get the GPU operation name.

        Args:
            op: GPU operation

        Returns:
            Operation name (e.g., "morphogen.gpu.launch") or None
        """
        if not MLIR_AVAILABLE:
            return None

        if not hasattr(op, "attributes"):
            return None

        op_name_attr = op.attributes.get("op_name")
        if op_name_attr is None:
            return None

        return str(op_name_attr).strip('"')


# Export public API
__all__ = [
    "GPUKernelType",
    "GPUBufferType",
    "GPUSharedType",
    "GPULaunchOp",
    "GPUParallelOp",
    "GPUAllocOp",
    "GPUAllocSharedOp",
    "GPUSyncOp",
    "GPUThreadIdOp",
    "GPUDialect",
    "MLIR_AVAILABLE",
]
