# Creative Computation DSL Examples

This directory contains example programs demonstrating various features of the Creative Computation DSL v0.2.2.

## 🎬 Interactive Python Examples (NEW!)

These examples showcase the new real-time interactive visualization:

### `interactive_diffusion.py`
Simple heat diffusion with live display. Perfect for getting started!
```bash
python examples/interactive_diffusion.py
```
**Features:** Real-time heat spreading, interactive controls, fire color palette

### `smoke_simulation.py`
Full Navier-Stokes fluid simulation with velocity and density fields.
```bash
python examples/smoke_simulation.py
```
**Features:** Incompressible flow, advection-diffusion-projection, swirling smoke patterns

### `reaction_diffusion.py`
Gray-Scott reaction-diffusion creating mesmerizing organic patterns.
```bash
python examples/reaction_diffusion.py
```
**Features:** Coral/maze patterns, self-organizing structures, stunning visuals

**Interactive Controls:**
- `SPACE` — Pause/Resume
- `→` — Step forward (when paused)
- `↑↓` — Adjust speed
- `Q/ESC` — Quit

## Directory Structure

- **fluids/** — Fluid dynamics and PDE-based simulations (DSL files)
- **agents/** — Agent-based and particle simulations (DSL files)
- **audio/** — Signal processing and audio synthesis (DSL files)
- **hybrid/** — Combined systems using multiple domains (DSL files)
- **Root directory** — Interactive Python examples (MVP-ready)

## Examples

### Fluid Dynamics

#### `fluids/navier_stokes.ccdsl`
Classic incompressible Navier-Stokes simulation for smoke and fluid effects. Demonstrates:
- Double-buffered field operations
- Advection-diffusion-projection pipeline
- Multiple solver methods (MacCormack advection, multigrid projection)
- Boundary conditions
- Parameter annotations with ranges and documentation

#### `fluids/reaction_diffusion.ccdsl`
Gray-Scott reaction-diffusion system generating organic patterns. Demonstrates:
- Coupled field evolution
- Laplacian operators
- Field combination operations
- Periodic boundary conditions
- Different parameter sets for various patterns

### Agent-Based Systems

#### `agents/boids.ccdsl`
Classic flocking behavior with separation, alignment, and cohesion rules. Demonstrates:
- Custom record types for agents
- Force-based agent interactions
- Spatial acceleration (grid method)
- Agent mapping and transformations
- Periodic boundary wrapping

### Audio Synthesis

#### `audio/fm_synthesis.ccdsl`
Frequency modulation synthesis with ADSR envelopes. Demonstrates:
- Signal domain operations
- Oscillators and waveform generation
- Signal mapping and transformation
- ADSR envelope generation
- Audio output

### Hybrid Systems

#### `hybrid/evolutionary_fluid.ccdsl`
Combines fluid dynamics with evolutionary agent-based simulation. Demonstrates:
- Integration of field and agent operations
- Agents sampling from fields with gradients
- Agent mutation and reproduction
- Adaptive timestep control
- Multi-layer visual composition

## Running Examples

To run an example program:

```bash
ccdsl run examples/fluids/navier_stokes.ccdsl
```

With custom parameters:

```bash
ccdsl run examples/fluids/navier_stokes.ccdsl --param viscosity=0.001
```

To validate without running:

```bash
ccdsl check examples/fluids/navier_stokes.ccdsl
```

## Learning Path

Recommended order for learning the DSL:

1. **Start with Fields** — `fluids/reaction_diffusion.ccdsl`
   - Understand field operations, double buffering, and basic PDE operations

2. **Agent Basics** — `agents/boids.ccdsl`
   - Learn agent types, force calculations, and spatial methods

3. **Signal Processing** — `audio/fm_synthesis.ccdsl`
   - Explore time-varying signals and audio synthesis

4. **Advanced Integration** — `fluids/navier_stokes.ccdsl`
   - Study complex solver configurations and method selection

5. **Hybrid Systems** — `hybrid/evolutionary_fluid.ccdsl`
   - Combine multiple domains and understand cross-domain operations

## Key Concepts Demonstrated

### Determinism
All examples are fully deterministic and will produce identical results across runs with the same seed.

### Profiles
Examples use different performance/precision profiles:
- `low` — Fast, reduced precision (audio, real-time)
- `medium` — Balanced (most simulations)
- `high` — Maximum precision (scientific computing)

### Solver Methods
Examples demonstrate various solver methods:
- **Advection:** Semi-Lagrangian, MacCormack
- **Diffusion:** Jacobi, Conjugate Gradient
- **Projection:** Jacobi, Multigrid
- **Integration:** Euler, Verlet

### Unit System
Examples show proper unit annotations:
- Velocity: `Vec2[m/s]`
- Position: `Vec2[m]`
- Frequency: `f32[Hz]`
- The type system enforces unit compatibility

## Extending Examples

Each example can be extended with:

1. **Different Solvers** — Try different `method` parameters
2. **Custom Functions** — Define your own `fn` for field/agent operations
3. **Additional Forces** — Add more force calculations for agents
4. **Visual Effects** — Experiment with different palettes and blend modes
5. **Benchmarking** — Add `@benchmark` decorators to measure performance

## Common Patterns

### Double Buffering
```dsl
@double_buffer state : Field2D<f32>
```
Automatically manages read/write buffers for in-place updates.

### Adaptive Timestep
```dsl
set dt = adaptive_dt(cfl=0.5, max_dt=0.02, min_dt=0.002)
```
Dynamically adjusts timestep for stability.

### Multi-stage Integration
```dsl
substep(4) {
  # Run 4 times with dt/4
}
```
Subdivide timesteps for accuracy.

### Field-Agent Coupling
```dsl
agents = agent.sample_field(agents, field, grad=true)
deposit = agent.deposit(agents, field, kernel="gaussian")
```
Bidirectional interaction between continuous fields and discrete agents.
