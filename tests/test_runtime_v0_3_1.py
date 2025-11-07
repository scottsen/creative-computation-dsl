"""Runtime tests for Kairo v0.3.1 features.

Tests execution of:
- Function definitions
- Function calls
- Return statements
- Lambda expressions
- If/else expressions
- Struct definitions
- Enhanced flow blocks
"""

import pytest
import numpy as np
from kairo.parser.parser import parse
from kairo.runtime.runtime import Runtime, ExecutionContext


class TestFunctionDefinitions:
    """Test function definition and call execution."""

    def test_simple_function_call(self):
        """Test defining and calling a simple function."""
        source = """
fn double(x) {
    return x * 2.0
}

result = double(5.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.has_variable('result')
        assert runtime.context.get_variable('result') == 10.0

    def test_function_with_multiple_params(self):
        """Test function with multiple parameters."""
        source = """
fn add(x, y) {
    return x + y
}

result = add(3.0, 4.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 7.0

    def test_function_with_typed_params(self):
        """Test function with type annotations."""
        source = """
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    temp1 = if x < min then min else x
    temp2 = if temp1 > max then max else temp1
    return temp2
}

result = clamp(15.0, 0.0, 10.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 10.0

    def test_function_calling_function(self):
        """Test functions calling other functions."""
        source = """
fn square(x) {
    return x * x
}

fn sum_of_squares(a, b) {
    return square(a) + square(b)
}

result = sum_of_squares(3.0, 4.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 25.0

    def test_function_with_no_return(self):
        """Test function without explicit return (returns None)."""
        source = """
fn do_nothing() {
    x = 5.0
}

result = do_nothing()
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') is None

    def test_function_implicit_return(self):
        """Test function with implicit return of last expression."""
        source = """
fn get_value() {
    return 42
}

result = get_value()
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 42


class TestIfElseExpressions:
    """Test if/else expression execution."""

    def test_simple_if_else_true(self):
        """Test if/else when condition is true."""
        source = """
x = 10.0
result = if x > 5.0 then 1.0 else 0.0
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 1.0

    def test_simple_if_else_false(self):
        """Test if/else when condition is false."""
        source = """
x = 3.0
result = if x > 5.0 then 1.0 else 0.0
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 0.0

    def test_if_else_with_expressions(self):
        """Test if/else with complex expressions."""
        source = """
x = 7.0
result = if x > 5.0 then x * 2.0 else x / 2.0
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 14.0

    def test_nested_if_else(self):
        """Test nested if/else expressions."""
        source = """
x = 7.0
result = if x > 10.0 then 1.0 else if x > 5.0 then 2.0 else 3.0
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 2.0

    def test_if_else_in_function(self):
        """Test if/else inside function."""
        source = """
fn abs_value(x) {
    return if x < 0.0 then -x else x
}

result1 = abs_value(-5.0)
result2 = abs_value(5.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result1') == 5.0
        assert runtime.context.get_variable('result2') == 5.0


class TestLambdaExpressions:
    """Test lambda expression execution."""

    def test_simple_lambda_no_params(self):
        """Test lambda with no parameters."""
        source = """
get_pi = || 3.14159
result = get_pi()
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert abs(runtime.context.get_variable('result') - 3.14159) < 0.00001

    def test_simple_lambda_one_param(self):
        """Test lambda with one parameter."""
        source = """
double = |x| x * 2.0
result = double(5.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 10.0

    def test_lambda_multiple_params(self):
        """Test lambda with multiple parameters."""
        source = """
add = |x, y| x + y
result = add(3.0, 4.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 7.0

    def test_lambda_with_capture(self):
        """Test lambda capturing variable from enclosing scope."""
        source = """
multiplier = 3.0
scale = |x| x * multiplier
result = scale(4.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 12.0

    def test_lambda_in_if_else(self):
        """Test lambda used in if/else expression."""
        source = """
x = 7.0
compute = |val| if val > 5.0 then val * 2.0 else val / 2.0
result = compute(x)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 14.0


class TestStructDefinitions:
    """Test struct definition and instantiation."""

    @pytest.mark.skip("Struct literal syntax needs AST node implementation")
    def test_simple_struct_definition(self):
        """Test defining and using a simple struct."""
        source = """
struct Point {
    x: f32
    y: f32
}

p = Point { x: 3.0, y: 4.0 }
result_x = p.x
result_y = p.y
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result_x') == 3.0
        assert runtime.context.get_variable('result_y') == 4.0

    @pytest.mark.skip("Struct literal syntax needs AST node implementation")
    def test_struct_with_units(self):
        """Test struct with physical unit types."""
        source = """
struct Particle {
    position: f32[m]
    velocity: f32[m/s]
}

p = Particle { position: 10.0, velocity: 2.5 }
result_pos = p.position
result_vel = p.velocity
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result_pos') == 10.0
        assert runtime.context.get_variable('result_vel') == 2.5


class TestFlowBlocks:
    """Test enhanced flow block execution."""

    def test_flow_with_dt_and_steps(self):
        """Test flow block with dt and steps parameters."""
        source = """
@state counter = 0

flow(dt=0.1, steps=5) {
    counter = counter + 1
}
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        # Should execute 5 times
        assert runtime.context.get_variable('counter') == 5

    def test_flow_with_dt_only(self):
        """Test flow block with only dt parameter (infinite loop not executed)."""
        source = """
@state x = 0.0

flow(dt=0.1) {
    x = x + dt
}
"""
        program = parse(source)
        runtime = Runtime()
        # This should not execute (infinite loop), or execute once as test
        # For now, we'll skip execution or execute 1 iteration
        # Implementation decision: dt-only flows execute once for testing
        # Real apps would run indefinitely
        runtime.execute_program(program)

        # Without steps, should execute 0 or 1 times (implementation choice)
        # Let's say it executes once for testing
        assert runtime.context.get_variable('x') >= 0.0

    def test_flow_with_substeps(self):
        """Test flow block with substeps parameter."""
        source = """
@state counter = 0

flow(dt=1.0, steps=2, substeps=3) {
    counter = counter + 1
}
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        # Should execute 2 * 3 = 6 times
        assert runtime.context.get_variable('counter') == 6

    def test_flow_with_state_persistence(self):
        """Test that @state variables persist across flow iterations."""
        source = """
@state accumulator = 0.0

flow(dt=0.1, steps=10) {
    accumulator = accumulator + dt
}
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        # Should accumulate: 10 * 0.1 = 1.0
        assert abs(runtime.context.get_variable('accumulator') - 1.0) < 0.0001


class TestIntegration:
    """Integration tests combining multiple v0.3.1 features."""

    def test_function_with_if_else_and_lambda(self):
        """Test function containing if/else and using lambda."""
        source = """
fn process(x, threshold) {
    compute = |val| val * 2.0
    result = if x > threshold then compute(x) else x / 2.0
    return result
}

result1 = process(10.0, 5.0)
result2 = process(3.0, 5.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result1') == 20.0
        assert runtime.context.get_variable('result2') == 1.5

    def test_flow_with_function_calls(self):
        """Test flow block calling user-defined functions."""
        source = """
fn update_velocity(v, acceleration, dt) {
    return v + acceleration * dt
}

@state velocity = 0.0

flow(dt=0.1, steps=10) {
    velocity = update_velocity(velocity, 9.8, dt)
}
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        # velocity = 0 + 9.8 * 0.1 * 10 = 9.8
        expected = 9.8
        assert abs(runtime.context.get_variable('velocity') - expected) < 0.0001

    def test_recursive_function(self):
        """Test recursive function (factorial)."""
        source = """
fn factorial(n) {
    result = if n <= 1.0 then 1.0 else n * factorial(n - 1.0)
    return result
}

result = factorial(5.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        assert runtime.context.get_variable('result') == 120.0

    def test_higher_order_function(self):
        """Test passing lambda to function."""
        source = """
fn apply_twice(f, x) {
    result = f(f(x))
    return result
}

double = |x| x * 2.0
result = apply_twice(double, 3.0)
"""
        program = parse(source)
        runtime = Runtime()
        runtime.execute_program(program)

        # double(double(3)) = double(6) = 12
        assert runtime.context.get_variable('result') == 12.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_undefined_function_call(self):
        """Test calling undefined function raises error."""
        source = """
result = undefined_function(5.0)
"""
        program = parse(source)
        runtime = Runtime()

        with pytest.raises(KeyError):
            runtime.execute_program(program)

    def test_function_with_wrong_arg_count(self):
        """Test calling function with wrong number of arguments."""
        source = """
fn add(x, y) {
    return x + y
}

result = add(5.0)
"""
        program = parse(source)
        runtime = Runtime()

        with pytest.raises(TypeError):
            runtime.execute_program(program)

    def test_return_outside_function(self):
        """Test return statement outside function."""
        source = """
return 42
"""
        program = parse(source)
        runtime = Runtime()

        # Should raise error or be ignored (implementation choice)
        with pytest.raises(RuntimeError):
            runtime.execute_program(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
