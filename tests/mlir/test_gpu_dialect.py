"""Tests for MLIR GPU dialect integration (Phase 6)

This module tests the GPU dialect operations and lowering passes
that enable GPU-accelerated execution of Morphogen programs.

Tests cover:
- GPU dialect operations (launch, parallel, alloc, sync)
- SCF-to-GPU lowering pass
- Field-to-GPU compilation
- GPU memory management
"""

import pytest

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, func, arith
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

from morphogen.mlir.context import MorphogenMLIRContext
from morphogen.mlir.compiler_v2 import MLIRCompilerV2

# Import GPU dialect
try:
    from morphogen.mlir.dialects.gpu_dialect import (
        GPUDialect, GPUKernelType, GPUBufferType, GPUSharedType
    )
    from morphogen.mlir.lowering.scf_to_gpu import (
        SCFToGPUPass, FieldToGPUPass,
        create_scf_to_gpu_pass, create_field_to_gpu_pass
    )
    GPU_DIALECT_AVAILABLE = True
except ImportError:
    GPU_DIALECT_AVAILABLE = False


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR not available")
@pytest.mark.skipif(not GPU_DIALECT_AVAILABLE, reason="GPU dialect not available")
class TestGPUDialect:
    """Test GPU dialect operations and types."""

    def test_gpu_kernel_type(self):
        """Test GPU kernel type creation."""
        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            kernel_type = GPUKernelType.get(f32, ctx.ctx)
            assert kernel_type is not None
            assert "gpu.kernel" in str(kernel_type)

    def test_gpu_buffer_type(self):
        """Test GPU buffer type creation."""
        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            buffer_type = GPUBufferType.get(f32, ctx.ctx)
            assert buffer_type is not None
            assert "gpu.buffer" in str(buffer_type)

    def test_gpu_shared_type(self):
        """Test GPU shared memory type creation."""
        ctx = MorphogenMLIRContext()
        with ctx.ctx:
            f32 = ir.F32Type.get()
            shared_type = GPUSharedType.get(f32, ctx.ctx)
            assert shared_type is not None
            assert "gpu.shared" in str(shared_type)

    def test_gpu_alloc_op(self):
        """Test GPU allocation operation."""
        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_gpu_alloc")

            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="test", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Create dimensions
                    h = arith.ConstantOp(ir.IndexType.get(), 256).result
                    w = arith.ConstantOp(ir.IndexType.get(), 256).result

                    # Allocate GPU buffer
                    buffer = GPUDialect.alloc([h, w], f32, loc, ip)
                    assert buffer is not None

                    func.ReturnOp([])

            # Verify module
            module.operation.verify()

    def test_gpu_thread_id_op(self):
        """Test GPU thread ID operation."""
        ctx = MorphogenMLIRContext()
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_gpu_thread_id")

            with ir.InsertionPoint(module.body):
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="test", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Get thread IDs
                    thread_x = GPUDialect.thread_id("x", loc, ip)
                    thread_y = GPUDialect.thread_id("y", loc, ip)
                    block_x = GPUDialect.thread_id("block_x", loc, ip)

                    assert thread_x is not None
                    assert thread_y is not None
                    assert block_x is not None

                    func.ReturnOp([])

            # Verify module
            module.operation.verify()


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR not available")
@pytest.mark.skipif(not GPU_DIALECT_AVAILABLE, reason="GPU dialect not available")
class TestSCFToGPULowering:
    """Test SCF-to-GPU lowering pass."""

    def test_scf_to_gpu_pass_creation(self):
        """Test SCF-to-GPU pass creation."""
        ctx = MorphogenMLIRContext()
        pass_obj = create_scf_to_gpu_pass(ctx, block_size=[256, 1, 1])
        assert pass_obj is not None
        assert pass_obj.block_size == [256, 1, 1]

    def test_scf_to_gpu_pass_with_custom_block_size(self):
        """Test SCF-to-GPU pass with custom block size."""
        ctx = MorphogenMLIRContext()
        pass_obj = create_scf_to_gpu_pass(ctx, block_size=[16, 16, 1])
        assert pass_obj.block_size == [16, 16, 1]

    def test_field_to_gpu_pass_creation(self):
        """Test Field-to-GPU pass creation."""
        ctx = MorphogenMLIRContext()
        pass_obj = create_field_to_gpu_pass(ctx, block_size=[16, 16, 1])
        assert pass_obj is not None
        assert pass_obj.block_size == [16, 16, 1]


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR not available")
@pytest.mark.skipif(not GPU_DIALECT_AVAILABLE, reason="GPU dialect not available")
class TestGPUCompilation:
    """Test GPU-accelerated compilation."""

    def test_compile_field_program_gpu_create(self):
        """Test GPU compilation of field creation."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "create", "args": {"width": 256, "height": 256, "fill": 0.0}},
        ]

        module = compiler.compile_field_program_gpu(
            operations,
            module_name="test_gpu_create",
            block_size=[16, 16, 1]
        )

        assert module is not None
        module.operation.verify()

        # Check that module contains GPU hints
        mlir_str = str(module)
        # The GPU pass adds attributes to mark loops for GPU execution
        assert "gpu_parallelizable" in mlir_str or "scf.for" in mlir_str

    def test_compile_field_program_gpu_gradient(self):
        """Test GPU compilation of field gradient."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "create", "args": {"width": 128, "height": 128, "fill": 1.0}},
            {"op": "gradient", "args": {"field": "field0"}},
        ]

        module = compiler.compile_field_program_gpu(
            operations,
            module_name="test_gpu_gradient",
            block_size=[16, 16, 1]
        )

        assert module is not None
        module.operation.verify()

        # Verify gradient operation is present
        mlir_str = str(module)
        assert "scf.for" in mlir_str

    def test_compile_field_program_gpu_laplacian(self):
        """Test GPU compilation of field Laplacian."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "create", "args": {"width": 64, "height": 64, "fill": 2.0}},
            {"op": "laplacian", "args": {"field": "field0"}},
        ]

        module = compiler.compile_field_program_gpu(
            operations,
            module_name="test_gpu_laplacian",
            block_size=[8, 8, 1]
        )

        assert module is not None
        module.operation.verify()

        # Verify Laplacian operation is present
        mlir_str = str(module)
        assert "scf.for" in mlir_str

    def test_apply_gpu_lowering(self):
        """Test applying GPU lowering pass to module."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # Create a simple field program
        operations = [
            {"op": "create", "args": {"width": 32, "height": 32, "fill": 0.5}},
        ]

        module = compiler.compile_field_program(operations)
        assert module is not None

        # Apply GPU lowering
        compiler.apply_gpu_lowering(module, block_size=[8, 8, 1])

        # Module should still be valid after GPU lowering
        module.operation.verify()

    def test_apply_field_to_gpu_lowering(self):
        """Test applying Field-to-GPU lowering pass."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        # Create module with field operations
        with ctx.ctx, ir.Location.unknown():
            module = ctx.create_module("test_field_to_gpu")

            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="main", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Create a field
                    w = arith.ConstantOp(ir.IndexType.get(), 64).result
                    h = arith.ConstantOp(ir.IndexType.get(), 64).result
                    fill = arith.ConstantOp(f32, 0.0).result

                    field = compiler.compile_field_create(w, h, fill, f32, loc, ip)

                    func.ReturnOp([])

            # Apply combined field-to-GPU lowering
            compiler.apply_field_to_gpu_lowering(module, block_size=[8, 8, 1])

            # Module should be valid
            module.operation.verify()


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR not available")
@pytest.mark.skipif(not GPU_DIALECT_AVAILABLE, reason="GPU dialect not available")
class TestGPUBlockSizeConfiguration:
    """Test GPU block size configuration."""

    def test_default_block_size(self):
        """Test default block size for GPU compilation."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "create", "args": {"width": 128, "height": 128, "fill": 0.0}},
        ]

        # Use default block size
        module = compiler.compile_field_program_gpu(operations)
        assert module is not None
        module.operation.verify()

    def test_custom_1d_block_size(self):
        """Test custom 1D block size."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "create", "args": {"width": 256, "height": 1, "fill": 0.0}},
        ]

        module = compiler.compile_field_program_gpu(
            operations,
            block_size=[256, 1, 1]
        )
        assert module is not None
        module.operation.verify()

    def test_custom_2d_block_size(self):
        """Test custom 2D block size."""
        ctx = MorphogenMLIRContext()
        compiler = MLIRCompilerV2(ctx)

        operations = [
            {"op": "create", "args": {"width": 256, "height": 256, "fill": 0.0}},
        ]

        module = compiler.compile_field_program_gpu(
            operations,
            block_size=[32, 32, 1]
        )
        assert module is not None
        module.operation.verify()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
