"""MLIR Compiler v2 for Kairo (v0.7.0)

This module implements the new MLIR-based compiler for Kairo, replacing
the text-based IR generation from v0.6.0 with real MLIR Python bindings.

Status: Phase 2 - Field Operations Dialect (Months 4-6)

Architecture:
    Kairo AST → MLIR IR (real bindings) → Lowering Passes → LLVM → Native Code

This is a complete rewrite of kairo/mlir/compiler.py to use actual MLIR
instead of string templates.

Phase 2 Additions:
- Field operations compilation
- FieldToSCF lowering pass integration
- Support for kairo.field.* operations
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from ..ast.nodes import (
    Program, Statement, Expression,
    Function, Return, Assignment, Literal, Identifier, BinaryOp
)

if TYPE_CHECKING:
    from .context import KairoMLIRContext

# Import MLIR if available
try:
    from mlir import ir
    from mlir.dialects import builtin, func, arith, scf, memref
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False
    if TYPE_CHECKING:
        from mlir import ir
    else:
        ir = None


class MLIRCompilerV2:
    """MLIR compiler using real Python bindings.

    This replaces the legacy text-based IR generation with actual MLIR
    IR construction using Python bindings.

    Example:
        >>> from kairo.mlir.context import KairoMLIRContext
        >>> ctx = KairoMLIRContext()
        >>> compiler = MLIRCompilerV2(ctx)
        >>> module = compiler.compile_program(ast_program)
    """

    def __init__(self, context: KairoMLIRContext):
        """Initialize MLIR compiler v2.

        Args:
            context: Kairo MLIR context

        Raises:
            RuntimeError: If MLIR bindings are not available
        """
        if not MLIR_AVAILABLE:
            raise RuntimeError(
                "MLIR Python bindings required but not installed. "
                "Install: pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
            )

        self.context = context
        self.module: Optional[Any] = None  # Will be ir.Module when MLIR is available
        self.symbols: Dict[str, Any] = {}  # Will be Dict[str, ir.Value] when MLIR is available

    def compile_program(self, program: Program) -> Any:
        """Compile a Kairo program to MLIR module.

        Args:
            program: Kairo Program AST node

        Returns:
            MLIR Module

        Status: TODO - Phase 1
        """
        raise NotImplementedError("Phase 1 implementation in progress")

    def compile_literal(self, literal: Literal, builder: Optional[Any]) -> Any:
        """Compile literal using arith.constant.

        Args:
            literal: Literal AST node
            builder: MLIR insertion point

        Returns:
            MLIR Value representing the constant

        Example:
            3.0 → %0 = arith.constant 3.0 : f32
        """
        with self.context.ctx:
            if isinstance(literal.value, float):
                f32 = ir.F32Type.get()
                return arith.ConstantOp(
                    f32,
                    ir.FloatAttr.get(f32, literal.value)
                ).result
            elif isinstance(literal.value, int):
                i32 = ir.I32Type.get()
                return arith.ConstantOp(
                    i32,
                    ir.IntegerAttr.get(i32, literal.value)
                ).result
            elif isinstance(literal.value, bool):
                i1 = ir.IntegerType.get_signless(1)
                return arith.ConstantOp(
                    i1,
                    ir.IntegerAttr.get(i1, 1 if literal.value else 0)
                ).result
            else:
                raise ValueError(f"Unsupported literal type: {type(literal.value)}")

    def compile_binary_op(self, binop: BinaryOp, builder: Optional[Any]) -> Any:
        """Compile binary operation.

        Args:
            binop: BinaryOp AST node
            builder: MLIR insertion point

        Returns:
            MLIR Value representing the result

        Example:
            x + y → %result = arith.addf %x, %y : f32

        Status: TODO - Phase 1
        """
        raise NotImplementedError("Phase 1 implementation in progress")

    # Phase 2: Field Operations Support

    def compile_field_create(
        self,
        width: Any,
        height: Any,
        fill_value: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile field creation operation.

        Args:
            width: Width dimension (ir.Value)
            height: Height dimension (ir.Value)
            fill_value: Initial fill value (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Field value

        Example:
            field.alloc((256, 256), fill_value=0.0)
            → %field = kairo.field.create %c256, %c256, %c0_f32 : !kairo.field<f32>
        """
        from .dialects.field import FieldDialect
        return FieldDialect.create(width, height, fill_value, element_type, loc, ip)

    def compile_field_gradient(
        self,
        field: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile gradient operation.

        Args:
            field: Input field (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Gradient field

        Example:
            field.gradient(field)
            → %grad = kairo.field.gradient %field : !kairo.field<f32>
        """
        from .dialects.field import FieldDialect
        return FieldDialect.gradient(field, loc, ip)

    def compile_field_laplacian(
        self,
        field: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile Laplacian operation.

        Args:
            field: Input field (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Laplacian field

        Example:
            field.laplacian(field)
            → %lapl = kairo.field.laplacian %field : !kairo.field<f32>
        """
        from .dialects.field import FieldDialect
        return FieldDialect.laplacian(field, loc, ip)

    def compile_field_diffuse(
        self,
        field: Any,
        rate: Any,
        dt: Any,
        iterations: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile diffusion operation.

        Args:
            field: Input field (ir.Value)
            rate: Diffusion rate (ir.Value)
            dt: Time step (ir.Value)
            iterations: Number of iterations (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Diffused field

        Example:
            field.diffuse(field, rate=0.1, dt=0.01, iterations=10)
            → %diffused = kairo.field.diffuse %field, %rate, %dt, %iters
        """
        from .dialects.field import FieldDialect
        return FieldDialect.diffuse(field, rate, dt, iterations, loc, ip)

    def apply_field_lowering(self, module: Any) -> None:
        """Apply field-to-SCF lowering pass to module.

        This transforms high-level field operations into low-level
        SCF loops and memref operations.

        Args:
            module: MLIR module to transform (in-place)

        Example:
            >>> compiler.apply_field_lowering(module)
            # Field ops → SCF loops + memref
        """
        from .lowering import create_field_to_scf_pass

        pass_obj = create_field_to_scf_pass(self.context)
        pass_obj.run(module)

    def compile_field_program(
        self,
        operations: List[Dict[str, Any]],
        module_name: str = "field_program"
    ) -> Any:
        """Compile a sequence of field operations to MLIR module.

        This is a convenience method for Phase 2 to compile field operations
        without requiring full AST support.

        Args:
            operations: List of operation dictionaries with keys:
                - op: Operation name ("create", "gradient", "laplacian", "diffuse")
                - args: Dictionary of arguments
            module_name: Module name

        Returns:
            MLIR Module with lowered operations

        Example:
            >>> ops = [
            ...     {"op": "create", "args": {"width": 256, "height": 256, "fill": 0.0}},
            ...     {"op": "gradient", "args": {"field": "field0"}},
            ... ]
            >>> module = compiler.compile_field_program(ops)
        """
        with self.context.ctx, ir.Location.unknown():
            module = self.context.create_module(module_name)

            # Create a wrapper function
            with ir.InsertionPoint(module.body):
                f32 = ir.F32Type.get()
                func_type = ir.FunctionType.get([], [])
                func_op = func.FuncOp(name="main", type=func_type)
                func_op.add_entry_block()

                with ir.InsertionPoint(func_op.entry_block):
                    loc = ir.Location.unknown()
                    ip = ir.InsertionPoint(func_op.entry_block)

                    # Process operations
                    results = {}
                    for i, operation in enumerate(operations):
                        op_name = operation["op"]
                        args = operation["args"]

                        if op_name == "create":
                            # Create constants for dimensions
                            width_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["width"])
                            ).result
                            height_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["height"])
                            ).result
                            fill_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["fill"])
                            ).result

                            result = self.compile_field_create(
                                width_val, height_val, fill_val, f32, loc, ip
                            )
                            results[f"field{i}"] = result

                        elif op_name == "gradient":
                            field_name = args["field"]
                            field_val = results[field_name]
                            result = self.compile_field_gradient(field_val, loc, ip)
                            results[f"grad{i}"] = result

                        elif op_name == "laplacian":
                            field_name = args["field"]
                            field_val = results[field_name]
                            result = self.compile_field_laplacian(field_val, loc, ip)
                            results[f"lapl{i}"] = result

                        elif op_name == "diffuse":
                            field_name = args["field"]
                            field_val = results[field_name]
                            rate_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["rate"])
                            ).result
                            dt_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["dt"])
                            ).result
                            iters_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["iterations"])
                            ).result

                            result = self.compile_field_diffuse(
                                field_val, rate_val, dt_val, iters_val, loc, ip
                            )
                            results[f"diffused{i}"] = result

                    # Return
                    func.ReturnOp([])

            # Apply lowering passes
            self.apply_field_lowering(module)

            return module


# Export for backward compatibility check
def is_legacy_compiler() -> bool:
    """Check if we're using the legacy text-based compiler.

    Returns:
        True if legacy compiler, False if using real MLIR
    """
    return not MLIR_AVAILABLE
