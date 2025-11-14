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

    # Phase 3: Temporal Operations Support

    def compile_flow_create(
        self,
        dt: Any,
        steps: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile flow creation operation.

        Args:
            dt: Time step size (ir.Value)
            steps: Number of timesteps (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Flow handle value

        Example:
            flow_create(dt=0.1, steps=10)
            → %flow = kairo.temporal.flow.create %dt, %steps : !kairo.flow<f32>
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.flow_create(dt, steps, element_type, loc, ip)

    def compile_flow_run(
        self,
        flow: Any,
        initial_state: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile flow run operation.

        Args:
            flow: Flow handle (ir.Value)
            initial_state: Initial state (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Final state value

        Example:
            flow_run(flow, initial_state)
            → %final = kairo.temporal.flow.run %flow, %initial_state
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.flow_run(flow, initial_state, None, loc, ip)

    def compile_state_create(
        self,
        size: Any,
        initial_value: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile state creation operation.

        Args:
            size: Size of state container (ir.Value)
            initial_value: Initial value (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            State container value

        Example:
            state_create(size=100, initial_value=0.0)
            → %state = kairo.temporal.state.create %size, %init_val : !kairo.state<f32>
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.state_create(size, initial_value, element_type, loc, ip)

    def compile_state_update(
        self,
        state: Any,
        index: Any,
        value: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile state update operation.

        Args:
            state: State container (ir.Value)
            index: Index to update (ir.Value)
            value: New value (ir.Value)
            loc: Source location
            ip: Insertion point

        Returns:
            Updated state container

        Example:
            state_update(state, index=5, value=1.5)
            → %new_state = kairo.temporal.state.update %state, %idx, %val
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.state_update(state, index, value, loc, ip)

    def compile_state_query(
        self,
        state: Any,
        index: Any,
        element_type: Any,
        loc: Any,
        ip: Any
    ) -> Any:
        """Compile state query operation.

        Args:
            state: State container (ir.Value)
            index: Index to read (ir.Value)
            element_type: MLIR element type
            loc: Source location
            ip: Insertion point

        Returns:
            Value at index

        Example:
            state_query(state, index=5)
            → %value = kairo.temporal.state.query %state, %idx : f32
        """
        from .dialects.temporal import TemporalDialect
        return TemporalDialect.state_query(state, index, element_type, loc, ip)

    def apply_temporal_lowering(self, module: Any) -> None:
        """Apply temporal-to-SCF lowering pass to module.

        This transforms high-level temporal operations into low-level
        SCF loops with memref-based state management.

        Args:
            module: MLIR module to transform (in-place)

        Example:
            >>> compiler.apply_temporal_lowering(module)
            # Temporal ops → SCF loops + memref
        """
        from .lowering import create_temporal_to_scf_pass

        pass_obj = create_temporal_to_scf_pass(self.context)
        pass_obj.run(module)

    def compile_temporal_program(
        self,
        operations: List[Dict[str, Any]],
        module_name: str = "temporal_program"
    ) -> Any:
        """Compile a sequence of temporal operations to MLIR module.

        This is a convenience method for Phase 3 to compile temporal operations
        without requiring full AST support.

        Args:
            operations: List of operation dictionaries with keys:
                - op: Operation name ("flow_create", "state_create", "flow_run", etc.)
                - args: Dictionary of arguments
            module_name: Module name

        Returns:
            MLIR Module with lowered operations

        Example:
            >>> ops = [
            ...     {"op": "state_create", "args": {"size": 100, "initial_value": 0.0}},
            ...     {"op": "flow_create", "args": {"dt": 0.1, "steps": 10}},
            ...     {"op": "flow_run", "args": {"flow": "flow0", "initial_state": "state0"}},
            ... ]
            >>> module = compiler.compile_temporal_program(ops)
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

                        if op_name == "state_create":
                            # Create constants
                            size_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["size"])
                            ).result
                            init_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["initial_value"])
                            ).result

                            result = self.compile_state_create(
                                size_val, init_val, f32, loc, ip
                            )
                            results[f"state{i}"] = result

                        elif op_name == "flow_create":
                            # Create constants
                            dt_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["dt"])
                            ).result
                            steps_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["steps"])
                            ).result

                            result = self.compile_flow_create(
                                dt_val, steps_val, f32, loc, ip
                            )
                            results[f"flow{i}"] = result

                        elif op_name == "flow_run":
                            flow_name = args["flow"]
                            state_name = args["initial_state"]
                            flow_val = results[flow_name]
                            state_val = results[state_name]

                            result = self.compile_flow_run(flow_val, state_val, loc, ip)
                            results[f"final_state{i}"] = result

                        elif op_name == "state_update":
                            state_name = args["state"]
                            state_val = results[state_name]
                            index_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["index"])
                            ).result
                            value_val = arith.ConstantOp(
                                f32,
                                ir.FloatAttr.get(f32, args["value"])
                            ).result

                            result = self.compile_state_update(
                                state_val, index_val, value_val, loc, ip
                            )
                            results[f"state{i}"] = result

                        elif op_name == "state_query":
                            state_name = args["state"]
                            state_val = results[state_name]
                            index_val = arith.ConstantOp(
                                ir.IndexType.get(),
                                ir.IntegerAttr.get(ir.IndexType.get(), args["index"])
                            ).result

                            result = self.compile_state_query(
                                state_val, index_val, f32, loc, ip
                            )
                            results[f"value{i}"] = result

                    # Return
                    func.ReturnOp([])

            # Apply lowering passes
            self.apply_temporal_lowering(module)

            return module


# Export for backward compatibility check
def is_legacy_compiler() -> bool:
    """Check if we're using the legacy text-based compiler.

    Returns:
        True if legacy compiler, False if using real MLIR
    """
    return not MLIR_AVAILABLE
