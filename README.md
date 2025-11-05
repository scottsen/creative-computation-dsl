# Creative Computation DSL v0.2.2

A typed, semantics-first domain-specific language for expressive, deterministic simulations and generative computation.

## Overview

Creative Computation DSL is a unified language where simulations, agents, signals, and visuals interoperate seamlessly — deterministically, portably, and joyfully.

### Key Features

- **Pure per-step graphs, explicit cross-step state** — Clear separation of computation within a timestep and state that persists across timesteps
- **Deterministic semantics** — Reproducible RNG (Philox 4×32-10) and stable ordering guarantees
- **Interactive visualization** — Real-time display with pause, step, and speed controls for immediate feedback
- **Composability + clarity** — Tiny vocabulary with maximal reuse across domains
- **MLIR-oriented lowering** — Every operation maps cleanly to MLIR dialects for efficient compilation
- **Live creativity** — Tunable solver profiles and hot-reload runtime for interactive development

## Language Domains

### Field Operations (PDE Toolkit)
Dense grid computations for fluid dynamics, reaction-diffusion, and physical simulations:
- Advection, diffusion, projection with multiple solver methods
- Stencil operations, gradients, and Laplacians
- Boundary conditions (reflect, periodic, noSlip)

### Agent-Based Systems
Sparse particle systems with deterministic evolution:
- Force calculations (including Barnes-Hut)
- Field sampling with gradients
- Mutation and reproduction for evolutionary algorithms
- Stable ordering by (id, creation_index)

### Signal Processing
Time-varying signals and audio synthesis:
- Oscillators, noise generators, ADSR envelopes
- Filters (1-pole, biquad)
- Integration, delays, and mixing
- Block-based rendering for audio output

### Visual Domain
Composable rendering pipeline:
- Field colorization with palettes
- Agent rendering as point sprites
- Layer composition with blend modes
- Post-processing filters and coordinate warps
- All operations in linear RGB

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tia-projects

# Install dependencies
pip install -e .
```

### Example: Evolutionary Fluid Hybrid

```dsl
set profile = medium
set dt = adaptive_dt(cfl=0.5, max_dt=0.02, min_dt=0.002)

@double_buffer vel, temp : Field2D<f32>
agents = step.state(agent.alloc(Particle, count=2000))

vel = field.advect(vel, vel, dt)
vel = field.project(vel, method="cg", iter=40)

temp = field.diffuse(temp, rate=κ, dt)
temp = field.react(temp, vel, Params{k:0.3})

agents = agent.sample_field(agents, temp, grad=true)
agents = agent.mutate(agents, fn=mutate_energy, rate=0.05)
agents = agent.reproduce(agents, template=default, rate=0.02)

visual.output( visual.layer([
  visual.colorize(temp, palette="fire"),
  visual.points(agents, color="white")
]) )
```

## Interactive Visualization (NEW! ✨)

CCDSL now features real-time interactive visualization! Watch your simulations come alive with smooth playback and full control.

### Quick Example

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual

def heat_diffusion():
    """Generate frames showing heat spreading."""
    temp = field.random((128, 128), seed=42, low=0.0, high=1.0)

    while True:
        temp = field.diffuse(temp, rate=0.2, dt=0.1, iterations=20)
        yield visual.colorize(temp, palette="fire")

# Display interactively
gen = heat_diffusion()
visual.display(lambda: next(gen), title="Heat Diffusion", target_fps=30, scale=4)
```

### Controls
- **SPACE**: Pause/Resume
- **→**: Step forward one frame (when paused)
- **↑↓**: Adjust speed
- **Q/ESC**: Quit

### Try the Examples

```bash
# Interactive heat diffusion
python examples/interactive_diffusion.py

# Stunning smoke simulation (Navier-Stokes)
python examples/smoke_simulation.py

# Mesmerizing reaction-diffusion patterns (Gray-Scott)
python examples/reaction_diffusion.py
```

See [Interactive Visualization Guide](docs/INTERACTIVE_VISUALIZATION.md) for full details.

## Documentation

### 📖 Start Here

**New to CCDSL?** Start with the [Complete Specification](SPECIFICATION.md) — a comprehensive guide covering everything from basics to advanced features with detailed examples.

### 🎓 Learning Path

1. **Introduction** → Read [SPECIFICATION.md](SPECIFICATION.md) sections 1-3 for overview and getting started
2. **Choose your domain** → Pick a section based on your interest:
   - Fields/PDEs → Section 6 + [examples/fluids/](examples/fluids/)
   - Agents/Particles → Section 7 + [examples/agents/](examples/agents/)
   - Audio/Signals → Section 8 + [examples/audio/](examples/audio/)
   - Multi-domain → Section 17 + [examples/hybrid/](examples/hybrid/)
3. **Deep dive** → Read sections 11-13 for determinism, solvers, and performance
4. **Reference** → Use [LANGUAGE_REFERENCE.md](LANGUAGE_REFERENCE.md) as quick lookup

### 📚 Documentation Structure

1. **[SPECIFICATION.md](SPECIFICATION.md)** — **Complete specification and tutorial**
   - Comprehensive guide with detailed explanations
   - Step-by-step tutorials and complete examples
   - Design principles and best practices
   - Performance tuning and optimization
   - **Start here if you're learning CCDSL**

2. **[LANGUAGE_REFERENCE.md](LANGUAGE_REFERENCE.md)** — **Quick reference**
   - Concise syntax and operator reference
   - Type system overview
   - Operation signatures and parameters
   - **Use this as a quick lookup**

3. **[examples/](examples/)** — **Runnable examples**
   - Complete working programs
   - Domain-specific examples (fluids, agents, audio, hybrid)
   - Example-specific README with learning path
   - **Browse these to see CCDSL in action**

4. **[docs/architecture.md](docs/architecture.md)** — **Implementation details**
   - Compiler architecture and MLIR lowering
   - Runtime system design
   - Developer documentation
   - **Read this if you're contributing to CCDSL**

## Project Structure

```
tia-projects/
├── SPECIFICATION.md          # Complete specification and tutorial (START HERE)
├── LANGUAGE_REFERENCE.md     # Quick reference for syntax and operators
├── README.md                 # This file
├── MVP.md                    # MVP definition and implementation plan
├── ROADMAP.md                # Development roadmap (v0.2.2 → v1.0.0)
├── STATUS.md                 # Current implementation status
├── LICENSE                   # MIT License
├── setup.py                  # Python package configuration
├── pyproject.toml            # Modern Python packaging
├── creative_computation/     # Main package
│   ├── __init__.py
│   ├── ast/                  # Abstract syntax tree definitions
│   ├── lexer/                # Lexical analysis
│   ├── parser/               # Syntax analysis
│   ├── types/                # Type system and unit checking
│   ├── mlir/                 # MLIR lowering
│   ├── runtime/              # Runtime execution engine
│   ├── stdlib/               # Standard library implementations
│   └── cli.py                # Command-line interface
├── examples/                 # Example programs
│   ├── README.md             # Example documentation and learning path
│   ├── fluids/               # Fluid dynamics examples
│   ├── agents/               # Agent-based examples
│   ├── audio/                # Signal processing examples
│   └── hybrid/               # Multi-domain examples
├── tests/                    # Test suite
│   ├── test_lexer.py
│   ├── test_parser.py
│   └── ...
└── docs/                     # Additional documentation
    └── architecture.md       # Implementation details
```

## Development Status

**Current Version:** v0.2.2-alpha
**Status:** Foundation complete, MVP in progress

### 🎯 Current Phase: MVP Implementation

We're currently implementing the Minimum Viable Product (MVP) focused on field operations (PDE toolkit):

**✅ Completed:**
- Language specification and comprehensive documentation
- Lexer and parser (full AST generation)
- Type system with physical units
- Type checker with error reporting
- Project structure and packaging

**🚧 In Progress:**
- Runtime execution engine
- Field operations (NumPy-based)
- Visualization pipeline (Pygame)

**📋 Next Steps:**
- Complete field PDE operations
- Get first examples running
- Cross-platform testing

For detailed status, see:
- **[STATUS.md](STATUS.md)** — What's implemented vs what needs to be done
- **[MVP.md](MVP.md)** — MVP definition and success criteria
- **[ROADMAP.md](ROADMAP.md)** — Long-term development roadmap

### 🎯 Target Examples for MVP

1. **Simple Diffusion** — Heat equation with colorful visualization
2. **Smoke Simulation** — Classic fluid dynamics (Navier-Stokes)
3. **Reaction-Diffusion** — Pattern formation (Gray-Scott)

### v0.2.2 Language Features

The specification includes all these features (implementation in progress):

- **Structure:** `iterate` for dynamic loops, `link` for graph visualization
- **Field:** `stencil`, `sample_grad`, `integrate` for richer PDE operations
- **Agent:** `mutate`, `reproduce` for evolutionary systems
- **Signal/Audio:** `block`, `io.output(audio)` for streaming DSP
- **Diagnostics:** `@benchmark`, `visual.tag`, `@metadata` for profiling

## Contributing

**We need your help!** This is an active open-source project looking for contributors.

### 🚀 How to Contribute

**Priority areas for MVP:**
1. **Runtime Engine** — Core execution loop and expression evaluation
2. **Field Operations** — NumPy-based PDE solvers (advection, diffusion, projection)
3. **Visualization** — Pygame-based display window
4. **Testing** — Unit tests and integration tests
5. **Documentation** — Getting started guide and tutorials

**Getting Started:**
1. Read [STATUS.md](STATUS.md) to see what needs work
2. Check [MVP.md](MVP.md) for detailed task breakdown
3. Pick a task that interests you
4. Open a GitHub issue to discuss your approach
5. Submit a PR with tests and documentation

**Good First Issues:**
- Field colorization with matplotlib colormaps
- Basic element-wise operations (map, combine)
- Unit tests for existing code
- Documentation improvements

**For detailed contribution guidelines, see:** [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon)

### 💬 Get in Touch

- **Issues:** Report bugs or request features on GitHub Issues
- **Discussions:** Ask questions or share ideas in GitHub Discussions
- **Email:** [project email - to be added]

We welcome contributions of all kinds: code, documentation, examples, bug reports, and feedback!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with inspiration from modern PDE solvers, creative coding frameworks, and the MLIR compiler infrastructure.
