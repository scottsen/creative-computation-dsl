"""Integration tests for end-to-end DSL program execution."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
from creative_computation.stdlib.field import field, Field2D
from creative_computation.stdlib.visual import visual
from creative_computation.runtime.runtime import ExecutionContext, Runtime


@pytest.mark.integration
class TestSimpleProgramExecution:
    """Integration tests for simple complete programs."""

    def test_heat_diffusion_pipeline(self):
        """Test complete heat diffusion pipeline."""
        # Create initial temperature field
        temp = field.random((64, 64), seed=42, low=0.0, high=1.0)
        assert temp.shape == (64, 64)

        # Apply diffusion
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
        assert temp.shape == (64, 64)

        # Apply boundary conditions
        temp = field.boundary(temp, spec="reflect")
        assert temp.shape == (64, 64)

        # Visualize
        vis = visual.colorize(temp, palette="fire")
        assert vis.shape == (64, 64)

        # Output to file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 1000
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_reaction_diffusion_pipeline(self):
        """Test reaction-diffusion pattern formation."""
        # Initialize fields
        u = field.random((128, 128), seed=1, low=0.0, high=1.0)
        v = field.random((128, 128), seed=2, low=0.0, high=1.0)

        # Simulate a few steps
        for _ in range(5):
            u = field.diffuse(u, rate=0.2, dt=0.1, iterations=10)
            v = field.diffuse(v, rate=0.1, dt=0.1, iterations=10)

            # Simple reaction (combine fields)
            reaction = field.combine(u, v, operation="mul")
            u = field.combine(u, reaction, operation="sub")

        # Visualize result
        vis = visual.colorize(u, palette="viridis")
        assert vis.shape == (128, 128)

    def test_velocity_field_projection(self):
        """Test velocity field with projection."""
        # Create divergent velocity field
        vx = field.random((32, 32), seed=1, low=-1.0, high=1.0)
        vy = field.random((32, 32), seed=2, low=-1.0, high=1.0)

        # Stack into velocity field
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Project to make divergence-free
        velocity = field.project(velocity, iterations=30)
        assert velocity.data.shape == (32, 32, 2)

        # Visualize magnitude (colorize computes magnitude for vector fields)
        vis = visual.colorize(velocity, palette="coolwarm")
        # Visual output is always 2D
        assert vis.shape == (32, 32)
        assert vis.data.shape == (32, 32, 3)  # RGB channels


@pytest.mark.integration
class TestRuntimeExecution:
    """Integration tests for runtime execution."""

    def test_context_with_multiple_operations(self):
        """Test execution context manages multiple operations."""
        ctx = ExecutionContext(global_seed=42)
        runtime = Runtime(ctx)

        # Create and store multiple fields
        temp = field.random((32, 32), seed=42)
        runtime.context.set_variable('temp', temp)

        velocity = Field2D(np.zeros((32, 32, 2)))
        runtime.context.set_variable('velocity', velocity)

        # Process multiple timesteps
        for step in range(5):
            temp = runtime.context.get_variable('temp')
            temp = field.diffuse(temp, rate=0.1, dt=0.01, iterations=5)
            runtime.context.set_variable('temp', temp)
            ctx.advance_timestep()

        # Verify state
        assert ctx.timestep == 5
        assert runtime.context.has_variable('temp')
        assert runtime.context.has_variable('velocity')

    def test_deterministic_execution(self):
        """Test that execution is deterministic."""
        def run_simulation(seed):
            ctx = ExecutionContext(global_seed=seed)
            runtime = Runtime(ctx)

            temp = field.random((64, 64), seed=seed)
            runtime.context.set_variable('temp', temp)

            for _ in range(10):
                temp = runtime.context.get_variable('temp')
                temp = field.diffuse(temp, rate=0.2, dt=0.1, iterations=10)
                runtime.context.set_variable('temp', temp)

            return runtime.context.get_variable('temp').data

        # Run twice with same seed
        result1 = run_simulation(12345)
        result2 = run_simulation(12345)

        # Should be identical
        assert np.array_equal(result1, result2)

        # Run with different seed
        result3 = run_simulation(54321)

        # Should be different
        assert not np.array_equal(result1, result3)


@pytest.mark.integration
class TestFieldOperationChains:
    """Integration tests for chaining field operations."""

    def test_long_operation_chain(self):
        """Test long chain of field operations."""
        f = field.random((64, 64), seed=42)

        # Chain many operations
        f = field.diffuse(f, rate=0.1, dt=0.01, iterations=5)
        f = field.boundary(f, spec="reflect")
        f = field.map(f, func="abs")
        f = field.diffuse(f, rate=0.2, dt=0.01, iterations=5)

        # Combine with another field
        f2 = field.alloc((64, 64), fill_value=0.5)
        f = field.combine(f, f2, operation="mul")

        # More processing
        f = field.diffuse(f, rate=0.1, dt=0.01, iterations=5)
        f = field.boundary(f, spec="periodic")

        # Should still have correct shape
        assert f.shape == (64, 64)
        assert not np.any(np.isnan(f.data))
        assert not np.any(np.isinf(f.data))

    def test_advection_diffusion_chain(self):
        """Test combined advection and diffusion."""
        # Create scalar field
        scalar = field.random((64, 64), seed=1)

        # Create velocity field
        vx = field.alloc((64, 64), fill_value=0.5)
        vy = field.alloc((64, 64), fill_value=0.5)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Simulate multiple steps
        for _ in range(10):
            scalar = field.advect(scalar, velocity, dt=0.01)
            scalar = field.diffuse(scalar, rate=0.1, dt=0.01, iterations=5)
            scalar = field.boundary(scalar, spec="reflect")

        assert scalar.shape == (64, 64)
        assert not np.any(np.isnan(scalar.data))


@pytest.mark.integration
class TestVisualizationPipeline:
    """Integration tests for visualization pipeline."""

    def test_multiple_palettes(self):
        """Test visualizing same field with multiple palettes."""
        f = field.random((64, 64), seed=42)

        palettes = ["grayscale", "fire", "viridis", "coolwarm"]
        outputs = []

        for palette in palettes:
            vis = visual.colorize(f, palette=palette)
            assert vis.shape == (64, 64)

            with tempfile.NamedTemporaryFile(suffix=f"_{palette}.png", delete=False) as tmp:
                tmp_path = tmp.name
                outputs.append(tmp_path)
                visual.output(vis, path=tmp_path)

        try:
            # Verify all files exist and have different contents
            for i, path in enumerate(outputs):
                assert Path(path).exists()
                assert Path(path).stat().st_size > 1000

            # Different palettes should produce different images
            with open(outputs[0], 'rb') as f1, open(outputs[1], 'rb') as f2:
                assert f1.read() != f2.read()
        finally:
            for path in outputs:
                if Path(path).exists():
                    Path(path).unlink()

    def test_value_range_visualization(self):
        """Test visualizing fields with different value ranges."""
        # Test with different value ranges
        ranges = [
            (0.0, 1.0),
            (-1.0, 1.0),
            (100.0, 200.0),
            (-1000.0, -900.0)
        ]

        for low, high in ranges:
            f = field.random((32, 32), seed=42, low=low, high=high)
            vis = visual.colorize(f, palette="fire")

            assert vis.shape == (32, 32)
            assert np.all(vis.data >= 0.0)
            assert np.all(vis.data <= 1.0)


@pytest.mark.integration
class TestComplexScenarios:
    """Integration tests for complex multi-component scenarios."""

    def test_smoke_simulation_simplified(self):
        """Test simplified smoke simulation."""
        # Initialize velocity field
        vx = field.random((64, 64), seed=1, low=-0.5, high=0.5)
        vy = field.random((64, 64), seed=2, low=-0.5, high=0.5)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Initialize density
        density = field.random((64, 64), seed=3, low=0.0, high=1.0)

        # Simulate several steps
        for step in range(5):
            # Advect velocity
            velocity = field.advect(velocity, velocity, dt=0.01)

            # Project velocity (make divergence-free)
            velocity = field.project(velocity, iterations=20)

            # Advect density
            density = field.advect(density, velocity, dt=0.01)

            # Diffuse density
            density = field.diffuse(density, rate=0.01, dt=0.01, iterations=10)

        # Visualize result
        vis = visual.colorize(density, palette="fire")
        assert vis.shape == (64, 64)

        # Save output
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            assert Path(tmp_path).exists()
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_multi_field_interaction(self):
        """Test multiple fields interacting."""
        # Create three fields
        f1 = field.random((32, 32), seed=1, low=0.0, high=1.0)
        f2 = field.random((32, 32), seed=2, low=0.0, high=1.0)
        f3 = field.random((32, 32), seed=3, low=0.0, high=1.0)

        # Process in parallel
        f1 = field.diffuse(f1, rate=0.1, dt=0.01, iterations=5)
        f2 = field.diffuse(f2, rate=0.2, dt=0.01, iterations=5)
        f3 = field.diffuse(f3, rate=0.3, dt=0.01, iterations=5)

        # Combine
        temp1 = field.combine(f1, f2, operation="add")
        result = field.combine(temp1, f3, operation="add")

        # Normalize
        result = field.map(result, func=lambda x: x / 3.0)

        # Final processing
        result = field.diffuse(result, rate=0.1, dt=0.01, iterations=10)

        assert result.shape == (32, 32)
        assert np.all(result.data >= 0.0)


@pytest.mark.integration
@pytest.mark.determinism
class TestDeterminismIntegration:
    """Integration tests specifically for determinism."""

    def test_full_pipeline_determinism(self):
        """Test that full pipeline is deterministic."""
        def run_full_pipeline(seed):
            temp = field.random((64, 64), seed=seed)
            temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
            temp = field.boundary(temp, spec="reflect")

            vis = visual.colorize(temp, palette="fire")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                visual.output(vis, path=tmp_path)
                with open(tmp_path, 'rb') as f:
                    return f.read()
            finally:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()

        # Run twice with same seed
        output1 = run_full_pipeline(42)
        output2 = run_full_pipeline(42)

        # Image bytes should be identical
        assert output1 == output2

    def test_operation_order_matters(self):
        """Test that operation order affects results."""
        f = field.random((32, 32), seed=42, low=-1.0, high=1.0)

        # Order 1: diffuse then square
        f1 = field.diffuse(f, rate=0.3, dt=0.1, iterations=20)
        f1 = field.map(f1, func="square")

        # Order 2: square then diffuse
        f2 = field.map(f, func="square")
        f2 = field.diffuse(f2, rate=0.3, dt=0.1, iterations=20)

        # Results should be different (operations don't commute)
        # Squaring after diffusion vs before produces different results
        assert not np.allclose(f1.data, f2.data, rtol=0.01)
