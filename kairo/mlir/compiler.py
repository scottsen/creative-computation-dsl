"""MLIR Compiler for Kairo v0.3.1

This module implements the core MLIR compilation pipeline, transforming
Kairo AST nodes into MLIR IR for native code generation.
"""

from typing import Dict, List, Optional, Any, Union
from .ir_builder import (
    IRBuilder, IRValue, IRType, IRFunction, IRBlock, IROperation, IRModule
)

from ..ast.nodes import (
    Program, Statement, Expression,
    Function, Return, Assignment, Flow, Struct, ExpressionStatement,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess,
    IfElse, Lambda, StructLiteral, Tuple,
    TypeAnnotation
)


class MLIRCompiler:
    """Compiles Kairo AST to MLIR IR.

    This compiler transforms Kairo programs into MLIR's multi-level intermediate
    representation, enabling progressive lowering to native machine code.

    Compilation Strategy:
    - SSA (Single Static Assignment) for immutable values
    - memref for state variables in flow blocks
    - Progressive lowering: high-level dialects → LLVM dialect → machine code

    Supported Phases:
    - Phase 1: Basic operations, functions, arithmetic ✅
    - Phase 2: Control flow (if/else), structs
    - Phase 3: Temporal execution (flow blocks)
    - Phase 4: Advanced features (lambdas, recursion)
    """

    def __init__(self):
        """Initialize MLIR compiler."""
        self.builder = IRBuilder()

        # Symbol tables
        self.symbols: Dict[str, IRValue] = {}  # Variable name → SSA value
        self.functions: Dict[str, IRFunction] = {}  # Function name → IR function
        self.struct_types: Dict[str, Dict[str, Any]] = {}  # Struct metadata

        # State tracking
        self.current_function: Optional[str] = None
        self.state_vars: Dict[str, Dict[str, Any]] = {}  # State variable metadata

        # Lambda counter for unique naming
        self.lambda_counter = 0

    def compile_program(self, program: Program) -> IRModule:
        """Compile a Kairo program to MLIR module.

        Args:
            program: Kairo Program AST node

        Returns:
            MLIR Module ready for lowering and execution

        Raises:
            ValueError: If compilation fails

        Note:
            Top-level code (assignments, expressions) is wrapped in a main() function.
            Function definitions are compiled as separate functions.
        """
        # Separate function definitions from other statements
        function_defs = []
        top_level_stmts = []

        for stmt in program.statements:
            if isinstance(stmt, Function):
                function_defs.append(stmt)
            else:
                top_level_stmts.append(stmt)

        # Compile function definitions first
        for func_def in function_defs:
            self.compile_function_def(func_def)

        # If there are top-level statements, wrap them in a main function
        if top_level_stmts:
            # Create implicit main function
            ir_func = self.builder.create_function(
                name="main",
                args=[],
                return_types=[]
            )
            self.functions["main"] = ir_func
            self.current_function = "main"

            # Create entry block
            self.builder.create_block(label="entry")

            # Compile top-level statements
            for stmt in top_level_stmts:
                self.compile_statement(stmt)

            # Add void return
            self.builder.add_operation(
                "func.return",
                operands=[],
                result_types=[]
            )

            self.current_function = None

        # Get and verify module
        module = self.builder.get_module()
        module.verify()

        return module

    def compile_statement(self, stmt: Statement) -> Optional[IRValue]:
        """Compile a statement.

        Args:
            stmt: Statement AST node

        Returns:
            MLIR operation (if any)
        """
        if isinstance(stmt, Function):
            return self.compile_function_def(stmt)
        elif isinstance(stmt, Return):
            return self.compile_return(stmt)
        elif isinstance(stmt, Assignment):
            return self.compile_assignment(stmt)
        elif isinstance(stmt, Flow):
            return self.compile_flow_block(stmt)
        elif isinstance(stmt, Struct):
            return self.compile_struct_def(stmt)
        elif isinstance(stmt, ExpressionStatement):
            return self.compile_expression(stmt.expression)
        else:
            raise NotImplementedError(f"Statement type not yet implemented: {type(stmt).__name__}")

    def compile_expression(self, expr: Expression) -> IRValue:
        """Compile an expression to an MLIR value.

        Args:
            expr: Expression AST node

        Returns:
            MLIR SSA value
        """
        if isinstance(expr, Literal):
            return self.compile_literal(expr)
        elif isinstance(expr, Identifier):
            return self.compile_identifier(expr)
        elif isinstance(expr, BinaryOp):
            return self.compile_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self.compile_unary_op(expr)
        elif isinstance(expr, Call):
            return self.compile_call(expr)
        elif isinstance(expr, FieldAccess):
            return self.compile_field_access(expr)
        elif isinstance(expr, IfElse):
            return self.compile_if_else(expr)
        elif isinstance(expr, Lambda):
            return self.compile_lambda(expr)
        elif isinstance(expr, StructLiteral):
            return self.compile_struct_literal(expr)
        elif isinstance(expr, Tuple):
            return self.compile_tuple(expr)
        else:
            raise NotImplementedError(f"Expression type not yet implemented: {type(expr).__name__}")

    # =========================================================================
    # Type System
    # =========================================================================

    def lower_type(self, kairo_type: Optional[TypeAnnotation]) -> IRType:
        """Convert Kairo type to MLIR type.

        Args:
            kairo_type: Kairo type annotation (may be None)

        Returns:
            MLIR type

        Examples:
            f32 → F32Type
            f32[m] → F32Type (units stripped)
            i32 → IntegerType(32)
            bool → IntegerType(1)
        """
        if kairo_type is None:
            # Default to f32 for untyped expressions
            return IRType.F32

        # Extract base type (strip physical units)
        base_type = kairo_type.base_type.lower()

        # Map Kairo base types to MLIR types
        if base_type == 'f32':
            return IRType.F32
        elif base_type == 'f64':
            return IRType.F64
        elif base_type == 'i32':
            return IRType.I32
        elif base_type == 'i64':
            return IRType.I64
        elif base_type == 'bool':
            return IRType.I1
        elif base_type in self.struct_types:
            # Struct type
            return self.struct_types[base_type]['mlir_type']
        else:
            # Unknown type, default to f32
            return IRType.F32

    def infer_type(self, expr: Expression) -> IRType:
        """Infer MLIR type from expression.

        Args:
            expr: Expression to infer type from

        Returns:
            Inferred MLIR type
        """
        if isinstance(expr, Literal):
            if isinstance(expr.value, float):
                return IRType.F32
            elif isinstance(expr.value, int):
                return IRType.I32
            elif isinstance(expr.value, bool):
                return IRType.I1
        elif isinstance(expr, Identifier):
            # Look up in symbol table and get type
            if expr.name in self.symbols:
                return self.symbols[expr.name].type
        elif isinstance(expr, BinaryOp):
            # Binary ops preserve type of operands (simplified)
            return self.infer_type(expr.left)

        # Default to f32
        return IRType.F32

    # =========================================================================
    # Phase 1.3: Literals and Identifiers
    # =========================================================================

    def compile_literal(self, literal: Literal) -> IRValue:
        """Compile a literal constant.

        Args:
            literal: Literal AST node

        Returns:
            MLIR constant value

        Examples:
            3.0 → arith.constant 3.0 : f32
            42 → arith.constant 42 : i32
            true → arith.constant 1 : i1
        """
        if isinstance(literal.value, float):
            # Float literal
            results = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.F32],
                attributes={"value": float(literal.value)}
            )
            return results[0]

        elif isinstance(literal.value, int):
            # Integer literal
            results = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.I32],
                attributes={"value": int(literal.value)}
            )
            return results[0]

        elif isinstance(literal.value, bool):
            # Boolean literal
            results = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.I1],
                attributes={"value": 1 if literal.value else 0}
            )
            return results[0]

        else:
            raise ValueError(f"Unsupported literal type: {type(literal.value)}")

    def compile_identifier(self, identifier: Identifier) -> IRValue:
        """Compile an identifier lookup.

        Args:
            identifier: Identifier AST node

        Returns:
            MLIR value from symbol table

        Raises:
            KeyError: If identifier is undefined
        """
        if identifier.name not in self.symbols:
            raise KeyError(f"Undefined variable: {identifier.name}")

        return self.symbols[identifier.name]

    # =========================================================================
    # Phase 1.4: Binary and Unary Operations
    # =========================================================================

    def compile_binary_op(self, binop: BinaryOp) -> IRValue:
        """Compile binary operation.

        Args:
            binop: BinaryOp AST node

        Returns:
            Result SSA value

        Examples:
            a + b → arith.addf %a, %b : f32
            x * y → arith.mulf %x, %y : f32
            i < j → arith.cmpf olt, %i, %j : f32
        """
        # Compile operands
        left = self.compile_expression(binop.left)
        right = self.compile_expression(binop.right)

        # Determine if operands are floating point or integer
        is_float = left.type in [IRType.F32, IRType.F64]

        # Map operator to MLIR operation
        if binop.operator == '+':
            opcode = "arith.addf" if is_float else "arith.addi"
        elif binop.operator == '-':
            opcode = "arith.subf" if is_float else "arith.subi"
        elif binop.operator == '*':
            opcode = "arith.mulf" if is_float else "arith.muli"
        elif binop.operator == '/':
            opcode = "arith.divf" if is_float else "arith.divsi"
        elif binop.operator == '%':
            opcode = "arith.remf" if is_float else "arith.remsi"
        elif binop.operator == '<':
            if is_float:
                opcode = "arith.cmpf"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "olt"}
                )
                return results[0]
            else:
                opcode = "arith.cmpi"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "slt"}
                )
                return results[0]
        elif binop.operator == '>':
            if is_float:
                opcode = "arith.cmpf"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "ogt"}
                )
                return results[0]
            else:
                opcode = "arith.cmpi"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "sgt"}
                )
                return results[0]
        elif binop.operator == '==':
            if is_float:
                opcode = "arith.cmpf"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "oeq"}
                )
                return results[0]
            else:
                opcode = "arith.cmpi"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "eq"}
                )
                return results[0]
        elif binop.operator == '!=':
            if is_float:
                opcode = "arith.cmpf"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "one"}
                )
                return results[0]
            else:
                opcode = "arith.cmpi"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "ne"}
                )
                return results[0]
        elif binop.operator == '<=':
            if is_float:
                opcode = "arith.cmpf"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "ole"}
                )
                return results[0]
            else:
                opcode = "arith.cmpi"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "sle"}
                )
                return results[0]
        elif binop.operator == '>=':
            if is_float:
                opcode = "arith.cmpf"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "oge"}
                )
                return results[0]
            else:
                opcode = "arith.cmpi"
                results = self.builder.add_operation(
                    opcode,
                    operands=[left, right],
                    result_types=[IRType.I1],
                    attributes={"predicate": "sge"}
                )
                return results[0]
        else:
            raise ValueError(f"Unsupported binary operator: {binop.operator}")

        # Perform operation for arithmetic ops
        results = self.builder.add_operation(
            opcode,
            operands=[left, right],
            result_types=[left.type]
        )
        return results[0]

    def compile_unary_op(self, unop: UnaryOp) -> IRValue:
        """Compile unary operation.

        Args:
            unop: UnaryOp AST node

        Returns:
            Result SSA value

        Examples:
            -x → arith.negf %x : f32
            !x → arith.xori %x, %true : i1
        """
        # Compile operand
        operand = self.compile_expression(unop.operand)

        if unop.operator == '-':
            # Negation
            is_float = operand.type in [IRType.F32, IRType.F64]
            if is_float:
                # For floats: 0.0 - x
                zero = self.builder.add_operation(
                    "arith.constant",
                    operands=[],
                    result_types=[operand.type],
                    attributes={"value": 0.0}
                )[0]
                results = self.builder.add_operation(
                    "arith.subf",
                    operands=[zero, operand],
                    result_types=[operand.type]
                )
            else:
                # For ints: 0 - x
                zero = self.builder.add_operation(
                    "arith.constant",
                    operands=[],
                    result_types=[operand.type],
                    attributes={"value": 0}
                )[0]
                results = self.builder.add_operation(
                    "arith.subi",
                    operands=[zero, operand],
                    result_types=[operand.type]
                )
            return results[0]

        elif unop.operator == '!':
            # Logical NOT: xor with 1
            one = self.builder.add_operation(
                "arith.constant",
                operands=[],
                result_types=[IRType.I1],
                attributes={"value": 1}
            )[0]
            results = self.builder.add_operation(
                "arith.xori",
                operands=[operand, one],
                result_types=[IRType.I1]
            )
            return results[0]

        else:
            raise ValueError(f"Unsupported unary operator: {unop.operator}")

    # =========================================================================
    # Phase 1.7: Assignments (SSA)
    # =========================================================================

    def compile_assignment(self, assign: Assignment) -> None:
        """Compile assignment statement (SSA style).

        Args:
            assign: Assignment AST node

        Note:
            In SSA, each variable gets a new value. The symbol table
            tracks the latest SSA value for each variable name.

        Example:
            x = 3.0 + 4.0
            → %0 = arith.constant 3.0 : f32
            → %1 = arith.constant 4.0 : f32
            → %x = arith.addf %0, %1 : f32
        """
        # Compile RHS expression
        value = self.compile_expression(assign.value)

        # Update symbol table (SSA: create new binding)
        self.symbols[assign.target] = value

    # =========================================================================
    # Phase 1.5: Function Definitions
    # =========================================================================

    def compile_function_def(self, func_node: Function) -> None:
        """Compile function definition.

        Args:
            func_node: Function AST node

        Example:
            fn add(x: f32, y: f32) -> f32 {
                return x + y
            }

            Becomes:
            func.func @add(%arg0: f32, %arg1: f32) -> f32 {
              %0 = arith.addf %arg0, %arg1 : f32
              func.return %0 : f32
            }
        """
        # Build function signature
        arg_types = []
        arg_values = []
        for i, (param_name, param_type) in enumerate(func_node.params):
            ir_type = self.lower_type(param_type)
            arg_types.append(ir_type)
            arg_value = IRValue(name=f"%arg{i}", type=ir_type)
            arg_values.append(arg_value)

        # Return types
        return_types = []
        if func_node.return_type:
            return_types.append(self.lower_type(func_node.return_type))

        # Create function
        ir_func = self.builder.create_function(
            name=func_node.name,
            args=arg_values,
            return_types=return_types
        )

        # Store function
        self.functions[func_node.name] = ir_func
        self.current_function = func_node.name

        # Create entry block
        self.builder.create_block(label="entry")

        # Save current symbol table
        saved_symbols = self.symbols.copy()

        # Map parameters to block arguments
        for (param_name, _), arg_value in zip(func_node.params, arg_values):
            self.symbols[param_name] = arg_value

        # Compile function body
        for stmt in func_node.body:
            if isinstance(stmt, Return):
                self.compile_return(stmt)
                break  # Return terminates the block
            else:
                self.compile_statement(stmt)

        # If no explicit return, add implicit void return
        if not func_node.body or not isinstance(func_node.body[-1], Return):
            self.builder.add_operation(
                "func.return",
                operands=[],
                result_types=[]
            )

        # Restore symbol table
        self.symbols = saved_symbols
        self.current_function = None

    def compile_return(self, return_node: Return) -> None:
        """Compile return statement.

        Args:
            return_node: Return AST node

        Example:
            return x + y
            → %0 = arith.addf %x, %y : f32
            → func.return %0 : f32
        """
        if return_node.value is not None:
            # Evaluate return value
            value = self.compile_expression(return_node.value)
            # Return with value
            self.builder.add_operation(
                "func.return",
                operands=[value],
                result_types=[]
            )
        else:
            # Void return
            self.builder.add_operation(
                "func.return",
                operands=[],
                result_types=[]
            )

    # =========================================================================
    # Phase 1.6: Function Calls
    # =========================================================================

    def compile_call(self, call: Call) -> IRValue:
        """Compile function call.

        Args:
            call: Call AST node

        Returns:
            Result value (or None for void functions)

        Example:
            result = add(3.0, 4.0)
            → %0 = arith.constant 3.0 : f32
            → %1 = arith.constant 4.0 : f32
            → %result = func.call @add(%0, %1) : (f32, f32) -> f32
        """
        # Get function name
        if isinstance(call.callee, Identifier):
            func_name = call.callee.name
        else:
            raise NotImplementedError("Only simple function calls supported in Phase 1")

        # Check if function exists
        if func_name not in self.functions:
            raise KeyError(f"Undefined function: {func_name}")

        # Compile arguments
        args = [self.compile_expression(arg) for arg in call.args]

        # Get function info
        ir_func = self.functions[func_name]

        # Create call operation
        if ir_func.return_types:
            # Function with return value
            results = self.builder.add_operation(
                "func.call",
                operands=args,
                result_types=ir_func.return_types,
                attributes={"callee": f"@{func_name}"}
            )
            return results[0]
        else:
            # Void function
            self.builder.add_operation(
                "func.call",
                operands=args,
                result_types=[],
                attributes={"callee": f"@{func_name}"}
            )
            # Return a dummy value (void functions don't have results)
            return None

    def compile_field_access(self, field_access: FieldAccess) -> IRValue:
        """Compile field access (Phase 2.4)."""
        raise NotImplementedError("Field access - Phase 2.4")

    def compile_if_else(self, if_else: IfElse) -> IRValue:
        """Compile if/else expression (Phase 2.1)."""
        raise NotImplementedError("If/else expressions - Phase 2.1")

    def compile_lambda(self, lambda_expr: Lambda) -> IRValue:
        """Compile lambda expression (Phase 4.1)."""
        raise NotImplementedError("Lambda expressions - Phase 4.1")

    def compile_struct_def(self, struct: Struct) -> None:
        """Compile struct definition (Phase 2.2)."""
        raise NotImplementedError("Struct definitions - Phase 2.2")

    def compile_struct_literal(self, struct_lit: StructLiteral) -> IRValue:
        """Compile struct literal (Phase 2.3)."""
        raise NotImplementedError("Struct literals - Phase 2.3")

    def compile_tuple(self, tuple_expr: Tuple) -> IRValue:
        """Compile tuple expression."""
        raise NotImplementedError("Tuple expressions")

    def compile_flow_block(self, flow: Flow) -> None:
        """Compile flow block (Phase 3.1)."""
        raise NotImplementedError("Flow blocks - Phase 3.1")
