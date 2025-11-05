"""Runtime execution engine for Creative Computation DSL.

This module provides the core runtime infrastructure for executing
Creative Computation DSL programs using NumPy as the backend.
"""

from typing import Any, Dict, Optional
import numpy as np


class ExecutionContext:
    """Manages execution state across timesteps.

    Handles:
    - Symbol table for variable storage
    - Double-buffered resources
    - Timestep management
    - Configuration settings
    """

    def __init__(self, global_seed: int = 42):
        """Initialize execution context.

        Args:
            global_seed: Global random seed for deterministic execution
        """
        self.symbols: Dict[str, Any] = {}
        self.double_buffers: Dict[str, tuple] = {}  # name -> (front, back)
        self.config: Dict[str, Any] = {}
        self.timestep: int = 0
        self.global_seed: int = global_seed
        self.dt: float = 0.01  # default timestep

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the symbol table.

        Args:
            name: Variable name
            value: Variable value
        """
        self.symbols[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable from the symbol table.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        if name not in self.symbols:
            raise KeyError(f"Undefined variable: {name}")
        return self.symbols[name]

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists.

        Args:
            name: Variable name

        Returns:
            True if variable exists
        """
        return name in self.symbols

    def register_double_buffer(self, name: str, front_buffer: Any, back_buffer: Any) -> None:
        """Register a double-buffered variable.

        Args:
            name: Variable name
            front_buffer: Front buffer (read from)
            back_buffer: Back buffer (write to)
        """
        self.double_buffers[name] = (front_buffer, back_buffer)
        self.symbols[name] = front_buffer

    def swap_buffers(self) -> None:
        """Swap double buffers at end of timestep."""
        for name, (front, back) in self.double_buffers.items():
            self.double_buffers[name] = (back, front)
            self.symbols[name] = back

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

        # Special handling for dt
        if key == "dt":
            self.dt = float(value)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def advance_timestep(self) -> None:
        """Advance to next timestep."""
        self.timestep += 1
        self.swap_buffers()


class Runtime:
    """Main runtime for executing Creative Computation DSL programs.

    Provides the interpreter that walks the AST and executes operations
    using NumPy-based implementations.
    """

    def __init__(self, context: Optional[ExecutionContext] = None):
        """Initialize runtime.

        Args:
            context: Execution context (creates new one if None)
        """
        self.context = context or ExecutionContext()
        self._setup_builtins()

    def _setup_builtins(self) -> None:
        """Set up built-in namespaces (field, visual, etc.)."""
        from ..stdlib.field import field
        from ..stdlib.visual import visual

        # Register built-in namespaces
        self.context.set_variable("field", field)
        self.context.set_variable("visual", visual)

    def execute_program(self, program) -> None:
        """Execute a complete DSL program.

        Args:
            program: Program AST node
        """
        from ..ast.nodes import Program

        if not isinstance(program, Program):
            raise TypeError(f"Expected Program node, got {type(program)}")

        # Execute statements in order
        for stmt in program.statements:
            self.execute_statement(stmt)

    def execute_statement(self, stmt) -> Any:
        """Execute a single statement.

        Args:
            stmt: Statement AST node

        Returns:
            Result of statement execution (if any)
        """
        from ..ast.nodes import (
            Assignment, ExpressionStatement, Step, Substep, Module, Compose,
            Call, Identifier, Literal, BinaryOp, UnaryOp, FieldAccess
        )

        # Handle different statement types
        if isinstance(stmt, Assignment):
            return self.execute_assignment(stmt)
        elif isinstance(stmt, ExpressionStatement):
            return self.execute_expression(stmt.expression)
        elif isinstance(stmt, Step):
            return self.execute_step(stmt)
        elif isinstance(stmt, Substep):
            return self.execute_substep(stmt)
        elif isinstance(stmt, Module):
            raise NotImplementedError("Module execution not yet implemented (post-MVP)")
        elif isinstance(stmt, Compose):
            raise NotImplementedError("Compose execution not yet implemented (post-MVP)")
        elif isinstance(stmt, Call):
            # Handle 'set' statements as special function calls
            if isinstance(stmt.func, Identifier) and stmt.func.name == "set":
                return self.execute_set_statement(stmt)
            return self.execute_expression(stmt)
        else:
            # Try to execute as expression
            return self.execute_expression(stmt)

    def execute_assignment(self, assign) -> None:
        """Execute an assignment statement.

        Args:
            assign: Assignment AST node
        """
        # Evaluate right-hand side
        value = self.execute_expression(assign.value)

        # Store in context (target is a string)
        self.context.set_variable(assign.target, value)

    def execute_set_statement(self, call) -> None:
        """Execute a 'set' configuration statement.

        Args:
            call: Call node representing 'set variable = value'
        """
        if len(call.args) != 1:
            raise ValueError("set statement requires exactly one argument")

        # Parse as assignment
        arg = call.args[0]
        from ..ast.nodes import BinaryOp

        if isinstance(arg, BinaryOp) and arg.op == "=":
            key = arg.left.name if hasattr(arg.left, 'name') else str(arg.left)
            value = self.execute_expression(arg.right)
            self.context.set_config(key, value)
        else:
            raise ValueError("set statement requires assignment syntax: set key = value")

    def execute_step(self, step) -> None:
        """Execute a step block.

        Args:
            step: Step AST node
        """
        # Execute all statements in the step
        for stmt in step.statements:
            self.execute_statement(stmt)

        # Advance timestep (swap buffers)
        self.context.advance_timestep()

    def execute_substep(self, substep) -> None:
        """Execute a substep block.

        Args:
            substep: Substep AST node
        """
        # Get iteration count
        n = self.execute_expression(substep.count)

        # Save original dt and divide by n
        original_dt = self.context.dt
        self.context.dt = original_dt / n

        # Execute n times
        for _ in range(n):
            for stmt in substep.statements:
                self.execute_statement(stmt)
            self.context.advance_timestep()

        # Restore original dt
        self.context.dt = original_dt

    def execute_expression(self, expr) -> Any:
        """Execute an expression and return its value.

        Args:
            expr: Expression AST node

        Returns:
            Evaluated expression value
        """
        from ..ast.nodes import (
            Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, Tuple
        )

        if isinstance(expr, Literal):
            return expr.value

        elif isinstance(expr, Identifier):
            return self.context.get_variable(expr.name)

        elif isinstance(expr, BinaryOp):
            return self.execute_binary_op(expr)

        elif isinstance(expr, UnaryOp):
            return self.execute_unary_op(expr)

        elif isinstance(expr, Call):
            return self.execute_call(expr)

        elif isinstance(expr, FieldAccess):
            return self.execute_field_access(expr)

        elif isinstance(expr, Tuple):
            # Evaluate all elements and return as Python tuple
            return tuple(self.execute_expression(elem) for elem in expr.elements)

        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def execute_binary_op(self, binop) -> Any:
        """Execute a binary operation.

        Args:
            binop: BinaryOp AST node

        Returns:
            Result of operation
        """
        left = self.execute_expression(binop.left)
        right = self.execute_expression(binop.right)

        # Map operators to NumPy operations
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '%': lambda a, b: a % b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
        }

        if binop.op not in ops:
            raise ValueError(f"Unknown binary operator: {binop.op}")

        return ops[binop.op](left, right)

    def execute_unary_op(self, unop) -> Any:
        """Execute a unary operation.

        Args:
            unop: UnaryOp AST node

        Returns:
            Result of operation
        """
        operand = self.execute_expression(unop.operand)

        if unop.op == '-':
            return -operand
        elif unop.op == '!':
            return not operand
        else:
            raise ValueError(f"Unknown unary operator: {unop.op}")

    def execute_call(self, call) -> Any:
        """Execute a function call.

        Args:
            call: Call AST node

        Returns:
            Result of function call
        """
        from ..ast.nodes import FieldAccess

        # Evaluate arguments
        args = [self.execute_expression(arg) for arg in call.args]
        kwargs = {k: self.execute_expression(v) for k, v in call.kwargs.items()}

        # Handle method calls (e.g., field.alloc(...))
        if isinstance(call.callee, FieldAccess):
            obj = self.execute_expression(call.callee.object)
            method_name = call.callee.field

            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return method(*args, **kwargs)
                else:
                    raise TypeError(f"'{method_name}' is not callable")
            else:
                raise AttributeError(f"Object has no method '{method_name}'")

        # Handle regular function calls
        func = self.execute_expression(call.callee)

        if callable(func):
            return func(*args, **kwargs)
        else:
            raise TypeError(f"Cannot call non-function: {type(func)}")

    def execute_field_access(self, field_access) -> Any:
        """Execute field access (method call or attribute access).

        Args:
            field_access: FieldAccess AST node

        Returns:
            Field or method
        """
        # Get the object
        obj = self.execute_expression(field_access.object)

        # Access the field
        if hasattr(obj, field_access.field):
            return getattr(obj, field_access.field)
        else:
            # Try as dictionary access
            if isinstance(obj, dict) and field_access.field in obj:
                return obj[field_access.field]

            raise AttributeError(f"Object has no field '{field_access.field}'")
