# Claude Context - Morphogen Project

## Project Overview

**Morphogen** (formerly Kairo) is a universal, deterministic computation platform that unifies multiple computational domains: audio synthesis, physics simulation, circuit design, geometry, and optimization—all in one type system, scheduler, and language.

- **Version**: 0.11.0
- **Status**: Production-Ready (40 computational domains)
- **Language**: Python-based runtime with MLIR compilation target
- **Philosophy**: Computation = Composition across domains

## Key Capabilities

### Production-Ready Domains (40+)
- **Audio/DSP**: Synthesis, filters, effects, physical modeling
- **Physics**: RigidBody dynamics, fluid simulation, field operations
- **Agents**: Particle systems, boids, N-body simulations
- **Graph**: Network analysis, path algorithms, community detection
- **Signal**: FFT, STFT, filtering, spectral analysis
- **Vision**: Edge detection, feature extraction, morphology
- **Chemistry**: Molecular dynamics, quantum chem, kinetics, thermodynamics
- **Procedural**: Noise, terrain, color palettes, image processing
- **Infrastructure**: Sparse linear algebra, integrators, I/O storage

### Cross-Domain Integration
All domains share:
- **Type system** with physical units (m, s, K, Hz, etc.)
- **Multirate scheduler** (audio @ 48kHz, control @ 60Hz, physics @ 240Hz)
- **MLIR compiler** (6 custom dialects → LLVM/GPU)
- **Deterministic execution** (bit-exact reproducibility)

## Project Structure

```
kairo/
├── src/morphogen/           # Core runtime and compiler
│   ├── runtime/             # Python interpreter
│   ├── compiler/            # MLIR compilation
│   ├── domains/             # Domain implementations (40+)
│   └── stdlib/              # Standard library
├── tests/                   # 900+ tests across 55 files
├── examples/                # Working examples for all domains
├── docs/                    # Comprehensive documentation
│   ├── architecture/        # System design
│   ├── specifications/      # 19 domain specs
│   ├── guides/              # Implementation guides
│   └── roadmap/             # Development roadmap
└── archive/                 # Historical docs

```

## Tools Available

### reveal - Progressive File Explorer

**Purpose**: Explore large files incrementally to manage token usage and understand structure before diving into full content.

**Install**:
```bash
# From gist: https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400
pip install reveal-cli
```

**Usage**:
```bash
# Level 0: Metadata only (filename, size, type, line count, hash)
reveal --level 0 src/morphogen/domains/audio.py

# Level 1: Structure (imports, classes, functions for Python files)
reveal --level 1 src/morphogen/domains/audio.py

# Level 2: Preview (representative sample with context)
reveal --level 2 docs/specifications/SPEC-AUDIO.md

# Level 3: Full content (with line numbers, optional paging)
reveal --level 3 --page-size 50 SPECIFICATION.md

# Grep filtering (works at any level)
reveal --level 1 --grep "class.*Analyzer" src/morphogen/compiler/
```

**When to Use reveal**:
- **Before reading large files**: Check structure at level 1 before committing to full read
- **Domain exploration**: Survey what's in a domain module (imports, classes, functions)
- **Documentation navigation**: Get markdown structure before reading 2000+ line specs
- **Large test files**: Preview test structure without loading all test code
- **Token conservation**: Get 80% of the information at 20% of the token cost

**File Type Support**:
- **Python**: AST analysis (imports, classes, functions, docstrings)
- **YAML**: Key structure, nesting depth, anchors/aliases
- **JSON**: Object/array counts, max depth, value types
- **Markdown**: Heading hierarchy, code blocks, lists
- **Text**: Generic line/word counts

## Working with Morphogen

### Common Tasks

**1. Adding a New Domain**
See `docs/guides/domain-implementation.md` for complete guide:
1. Create domain module in `src/morphogen/domains/`
2. Define operators (functions with type signatures)
3. Add to operator registry
4. Write tests in `tests/domains/`
5. Document in `docs/specifications/`

**2. Understanding Existing Domains**
```bash
# Get domain structure first
reveal --level 1 src/morphogen/domains/audio.py

# See what tests exist
reveal --level 1 tests/domains/test_audio.py

# Check specification
reveal --level 2 docs/specifications/SPEC-AUDIO.md
```

**3. Running Tests**
```bash
# All tests
pytest tests/

# Specific domain
pytest tests/domains/test_audio.py -v

# With coverage
pytest tests/ --cov=src/morphogen --cov-report=html
```

**4. Exploring Documentation**
Start with `docs/README.md` for navigation, then:
- `ARCHITECTURE.md` - System design
- `ECOSYSTEM_MAP.md` - All domains mapped
- `SPECIFICATION.md` - Language reference
- `docs/architecture/domain-architecture.md` - Deep domain specs

## Development Workflow

### Current Branch
**Branch**: `claude/document-tools-capabilities-01FMGmWoo35SWwRuXmLvKA63`

### Git Practices
```bash
# Always push to the claude/* branch
git add .
git commit -m "docs: Document reveal tool integration"
git push -u origin claude/document-tools-capabilities-01FMGmWoo35SWwRuXmLvKA63
```

### Before Making Changes
1. **Check structure** with reveal at level 1
2. **Read selectively** using Read tool for specific files
3. **Search strategically** with Grep for patterns
4. **Test changes** with pytest
5. **Document updates** in relevant specs/guides

## Key Architectural Concepts

### Temporal Model
Programs use `flow` blocks for time evolution:
```morphogen
@state temp : Field2D<f32 [K]> = zeros((128, 128))

flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
    output colorize(temp, palette="fire")
}
```

### Deterministic RNG
All randomness explicit via RNG objects:
```morphogen
@state agents : Agents<Particle> = alloc(count=100, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100))
    }
}
```

### Cross-Domain Composition
Domains work together seamlessly:
```morphogen
use fluid, acoustics, audio

@state flow : FluidNetwork1D = engine_exhaust(length=2.5m)
@state acoustic : AcousticField1D = waveguide_from_flow(flow)

flow(dt=0.1ms) {
    flow = flow.advance(engine_pulse(t))
    acoustic = acoustic.couple_from_fluid(flow)
    let sound = acoustic.to_audio(mic_position=1.5m)
    audio.play(sound)
}
```

## Strategic Context

### Professional Applications
- **Education**: Replace MATLAB for computational physics/engineering
- **Digital Twins**: Unified multi-physics simulation (automotive, aerospace)
- **Audio Production**: Physical modeling for virtual instruments
- **Scientific Computing**: Multi-domain research without tool fragmentation
- **Creative Coding**: Deterministic generative art + audio + physics

### Long-Term Vision
- GPU acceleration via MLIR GPU dialect
- JIT compilation for live performance
- Advanced optimizations (fusion, vectorization, polyhedral)
- Geometry/CAD integration (TiaCAD-inspired reference system)
- Symbolic math and control theory domains

## Sister Project: Philbrick

**Philbrick** is the analog/hybrid hardware counterpart to Morphogen (digital software). Both share the same four core operations (sum, integrate, nonlinearity, events) and compositional philosophy.

- **Morphogen**: Digital simulation of continuous phenomena
- **Philbrick**: Physical embodiment of continuous dynamics
- **Bridge**: Design in Morphogen → Build in Philbrick → Validate together

## Quick Reference

### Important Files
- `ARCHITECTURE.md` - System architecture overview
- `SPECIFICATION.md` - Complete language reference
- `AUDIO_SPECIFICATION.md` - Audio DSL reference
- `ECOSYSTEM_MAP.md` - All domains mapped
- `docs/architecture/domain-architecture.md` - Deep domain specs (2,266 lines)
- `docs/DOMAIN_VALUE_ANALYSIS.md` - Strategic analysis

### Running Examples
```bash
# Heat diffusion
python examples/field_ops/heat_diffusion.py

# Audio synthesis
python examples/audio/karplus_strong.py

# Rigid body physics
python examples/rigidbody_physics/bouncing_balls.py

# Agent simulation
python examples/agents/boids.py
```

### Test Coverage
- **900+ tests** across 55 test files
- Full domain coverage for all production domains
- Integration tests for cross-domain composition

---

**Last Updated**: 2025-11-21
**Status**: v0.11.0 - Complete Multi-Domain Platform (40 domains, 500+ operators)
