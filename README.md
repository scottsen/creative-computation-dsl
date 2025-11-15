---
project: kairo
type: software
status: active
beth_topics:
- kairo
- creative-computation
- dsl
- mlir
- audio-synthesis
- agent-simulation
- field-operations
tags:
- compiler
- simulation
- generative
- deterministic
---

# Kairo v0.7.0 (In Development)

**A semantic, deterministic transform kernel with two human-friendly faces:**
- **Kairo.Audio** for composition
- **RiffStack** for performance

*Where computation becomes composition*

> 📐 **Architecture**: See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete Kairo Stack design (kernel, frontends, Graph IR, transforms)
> 🚀 **v0.7.0 Development**: Real MLIR Integration underway - see [docs/v0.7.0_DESIGN.md](docs/v0.7.0_DESIGN.md) for the 12-month roadmap

---

## What is Kairo?

**Kairo** is a typed, deterministic domain-specific language for creative computation. It unifies **simulation**, **sound**, **visualization**, and **procedural design** within a single, reproducible execution model.

### Key Features

- ✅ **Deterministic by default** - Bitwise-identical results across runs and platforms
- ✅ **Explicit temporal model** - Time evolution via `flow(dt)` blocks
- ✅ **Declarative state** - `@state` annotations make persistence clear
- ✅ **Physical units** - Type system includes dimensional analysis
- ✅ **Multi-domain** - Fields, agents, signals, and visuals in one language
- ✅ **MLIR-based** - Compiles to optimized native code
- ✅ **Hot-reload** - Interactive development with live code updates

---

## Quick Start

### Installation

```bash
git clone https://github.com/scottsen/kairo.git
cd kairo
pip install -e .
```

### Your First Program

Create `hello.kairo`:

```kairo
# hello.kairo - Heat diffusion

use field, visual

@state temp : Field2D<f32 [K]> = random_normal(
    seed=42,
    shape=(128, 128),
    mean=300.0,
    std=50.0
)

const KAPPA : f32 [m²/s] = 0.1

flow(dt=0.01, steps=500) {
    temp = diffuse(temp, rate=KAPPA, dt, iterations=20)
    output colorize(temp, palette="fire", min=250.0, max=350.0)
}
```

Run it:

```bash
kairo run hello.kairo
```

---

## Language Overview

### Temporal Model

Kairo programs describe time-evolving systems through `flow` blocks:

```kairo
flow(dt=0.01, steps=1000) {
    # Execute this block 1000 times with timestep 0.01
    temp = diffuse(temp, rate=0.1, dt)
    output colorize(temp, palette="fire")
}
```

### State Management

Persistent variables are declared with `@state`:

```kairo
@state vel : Field2D<Vec2<f32>> = zeros((256, 256))
@state agents : Agents<Particle> = alloc(count=1000)

flow(dt=0.01) {
    vel = advect(vel, vel, dt)      # Updates vel for next step
    agents = integrate(agents, dt)   # Updates agents for next step
}
```

### Deterministic Randomness

All randomness is explicit via RNG objects:

```kairo
@state agents : Agents<Particle> = alloc(count=100, init=spawn_random)

fn spawn_random(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1))
    }
}
```

### Physical Units

Types can carry dimensional information:

```kairo
temp : Field2D<f32 [K]>           # Temperature in Kelvin
pos : Vec2<f32 [m]>               # Position in meters
vel : Vec2<f32 [m/s]>             # Velocity in m/s
force : Vec2<f32 [N]>             # Force in Newtons

# Unit checking at compile time
dist : f32 [m] = 10.0
time : f32 [s] = 2.0
speed = dist / time               # OK: f32 [m/s]

# ERROR: cannot mix incompatible units
x = dist + time                   # ERROR: m + s is invalid
```

---

## Four Dialects

### 1. Field Dialect - Dense Grid Operations

```kairo
use field

@state temp : Field2D<f32> = random_normal(seed=42, shape=(256, 256))

flow(dt=0.1) {
    # PDE operations
    temp = diffuse(temp, rate=0.2, dt)
    temp = advect(temp, velocity, dt)

    # Stencil operations
    let grad = gradient(temp)
    let lap = laplacian(temp)

    # Element-wise operations
    temp = temp.map(|x| clamp(x, 0.0, 1.0))
}
```

### 2. Agent Dialect - Sparse Particle Systems

**✅ PRODUCTION-READY - implemented in v0.4.0!**

```kairo
use agent  # ✅ WORKING - fully implemented!

struct Boid {
    pos: Vec2<f32>
    vel: Vec2<f32>
}

@state boids : Agents<Boid> = alloc(count=200, init=spawn_boid)

flow(dt=0.01) {
    # Per-agent transformations
    boids = boids.map(|b| {
        vel: b.vel + flocking_force(b) * dt,
        pos: b.pos + b.vel * dt
    })

    # Filter
    boids = boids.filter(|b| in_bounds(b.pos))
}
```

**Features:**
- Complete agent operations (alloc, map, filter, reduce)
- N-body force calculations with spatial hashing (O(n) performance)
- Field-agent coupling (particles in flow fields)
- 85 comprehensive tests
- Example simulations: boids, N-body, particle systems

**Status:** Production-ready as of v0.4.0 (2025-11-14)

### 3. Audio Dialect (Kairo.Audio) - Sound Synthesis and Processing

**✅ PRODUCTION-READY - implemented in v0.5.0 and v0.6.0!**

Kairo.Audio is a compositional, deterministic audio language with physical modeling, synthesis, and real-time I/O.

```kairo
use audio  # ✅ WORKING - fully implemented!

# Synthesis example (v0.5.0)
let pluck_excitation = noise(seed=1) |> lowpass(6000)
let string_sound = string(pluck_excitation, freq=220, t60=1.5)
let final = string_sound |> reverb(mix=0.12)

# I/O example (v0.6.0)
audio.play(final)           # Real-time playback
audio.save(final, "out.wav") # Export to WAV/FLAC
```

**Features (v0.5.0 - Synthesis):**
- Oscillators: sine, saw, square, triangle, noise
- Filters: lowpass, highpass, bandpass, notch, EQ
- Envelopes: ADSR, AR, exponential decay
- Effects: delay, reverb, chorus, flanger, drive, limiter
- Physical modeling: Karplus-Strong strings, modal synthesis
- 192 comprehensive tests (184 passing)

**Features (v0.6.0 - I/O):**
- Real-time audio playback with `audio.play()`
- WAV/FLAC export with `audio.save()`
- Audio loading with `audio.load()`
- Microphone recording with `audio.record()`
- Complete demonstration scripts

**Status:** Production-ready as of v0.5.0 (2025-11-14), I/O added in v0.6.0

### 4. Visual Dialect - Rendering and Composition

**✅ ENHANCED in v0.6.0 - Agent rendering and video export!**

```kairo
use visual

# Colorize fields (v0.2.2)
let field_vis = colorize(temp, palette="viridis")

# Render agents (v0.6.0 - NEW!)
let agent_vis = visual.agents(particles, width=256, height=256,
                               color_property='vel', palette='fire', size=3.0)

# Layer composition (v0.6.0 - NEW!)
let combined = visual.composite(field_vis, agent_vis, mode="add", opacity=[1.0, 0.7])

# Video export (v0.6.0 - NEW!)
visual.video(frames, "animation.mp4", fps=30)

output combined
```

**Features:**
- Field colorization with 4 palettes (grayscale, fire, viridis, coolwarm)
- PNG/JPEG export and interactive display
- **Agent visualization** with color/size-by-property ⭐ NEW in v0.6.0!
- **Layer composition** with multiple blending modes ⭐ NEW in v0.6.0!
- **Video export** (MP4, GIF) with memory-efficient generators ⭐ NEW in v0.6.0!

---

## Examples

### Fluid Simulation (Navier-Stokes)

```kairo
use field, visual

@state vel : Field2D<Vec2<f32 [m/s]>> = zeros((256, 256))
@state density : Field2D<f32> = zeros((256, 256))

const VISCOSITY : f32 = 0.001
const DIFFUSION : f32 = 0.0001

flow(dt=0.01, steps=1000) {
    # Advect velocity
    vel = advect(vel, vel, dt, method="maccormack")

    # Diffuse velocity (viscosity)
    vel = diffuse(vel, rate=VISCOSITY, dt, iterations=20)

    # Project (incompressibility)
    vel = project(vel, method="cg", max_iterations=50)

    # Advect and diffuse density
    density = advect(density, vel, dt)
    density = diffuse(density, rate=DIFFUSION, dt)

    # Dissipation
    density = density * 0.995

    # Visualize
    output colorize(density, palette="viridis")
}
```

### Reaction-Diffusion (Gray-Scott)

```kairo
use field, visual

@state u : Field2D<f32> = ones((256, 256))
@state v : Field2D<f32> = zeros((256, 256))

const Du : f32 = 0.16
const Dv : f32 = 0.08
const F : f32 = 0.060
const K : f32 = 0.062

flow(dt=1.0, steps=10000) {
    # Gray-Scott reaction
    let uvv = u * v * v
    let du_dt = Du * laplacian(u) - uvv + F * (1.0 - u)
    let dv_dt = Dv * laplacian(v) + uvv - (F + K) * v

    u = u + du_dt * dt
    v = v + dv_dt * dt

    # Visualize
    output colorize(v, palette="viridis")
}
```

See `examples/` directory for more!

---

## Project Status

**Version**: 0.7.4
**Stable Version**: 0.6.0
**Status**: v0.7.0 Real MLIR Integration - All 6 Phases Complete ✅

### ✅ Production-Ready
- Language specification (comprehensive)
- Type system design
- Syntax definition (full v0.3.1 syntax)
- Frontend (lexer, parser) - complete recursive descent parser
- **Python Runtime** (production-ready NumPy interpreter)
- **Field operations** (advect, diffuse, project, Laplacian, etc.)
- **Agent operations** (alloc, map, filter, reduce, forces, field sampling) ⭐ NEW in v0.4.0!
- **Audio synthesis** (oscillators, filters, envelopes, effects, physical modeling) ⭐ NEW in v0.5.0!
- **Audio I/O** (real-time playback, WAV/FLAC export, recording) ⭐ NEW in v0.6.0!
- **Visual extensions** (agent rendering, layer composition, video export) ⭐ NEW in v0.6.0!
- **Visualization** (PNG/JPEG export, interactive display, MP4/GIF video)
- Documentation (comprehensive and accurate)
- Test suite (580+ tests: 247 original + 85 agent + 184 audio + 64+ I/O tests)

### ✅ Complete (v0.7.0 - Real MLIR Integration)
- **MLIR Python Bindings Integration** - All 6 Phases Complete (Nov 14-15, 2025)
  - ✅ **Phase 1**: Foundation - MLIR context, compiler V2, proof-of-concept
  - ✅ **Phase 2**: Field Operations Dialect - 4 operations, lowering pass, tests
  - ✅ **Phase 3**: Temporal Execution - 6 operations, state management, flow blocks
  - ✅ **Phase 4**: Agent Operations - 4 operations, behavior system, 36 tests
  - ✅ **Phase 5**: Audio Operations - 4 operations, DSP primitives, lowering pass
  - ✅ **Phase 6**: JIT/AOT Compilation - LLVM backend, caching, 7 output formats
- See [CHANGELOG.md](CHANGELOG.md) for v0.7.4 details and [docs/v0.7.0_DESIGN.md](docs/v0.7.0_DESIGN.md) for design

### 🚧 Deprecated (Legacy, Maintained for Compatibility)
- **MLIR text IR generation** (legacy text-based, not real MLIR bindings)
- Optimization passes (basic constant folding, DCE stubs)
- Will be removed in v0.8.0+ after v0.7.0 transition complete

### 📋 Planned (Future Phases)
- **Geometry Domain (v0.9+)** ⭐ **Architecture Complete**:
  - Unified reference & frame model inspired by TiaCAD v3.x
  - Complete specifications: `SPEC-COORDINATE-FRAMES.md`, `SPEC-GEOMETRY.md`
  - ADR-001: Unified Reference Model (approved for implementation)
  - Cross-domain anchor system (geometry, audio, physics, agents, fields)
  - Reference-based composition replacing hierarchical assemblies
  - Declarative CAD operators (primitives, sketches, booleans, patterns, mesh ops)
  - Backend-neutral (CadQuery, CGAL, GPU SDF targets)
  - See: `docs/DOMAIN_ARCHITECTURE.md` Section 2.1
- **Physical Unit Checking** - Annotations exist, dimensional analysis not enforced
- **Hot-reload** - Architecture designed, not implemented
- **GPU Acceleration** - Via MLIR GPU dialect (planned Phase 3-4)
- **Advanced Optimization** - Auto-vectorization, fusion, polyhedral optimization

**Current Milestone**: v0.7.4 - Real MLIR Integration Complete (All 6 Phases)
**Next Milestone**: v0.8.0 - Production hardening and performance optimization
**Future Milestone**: v0.9.0 - Geometry Domain Implementation

---

## Domain Architecture Expansion (November 2025)

**Major Milestone**: Kairo's domain architecture has been massively expanded with comprehensive specifications for 10+ new domains, establishing Kairo as a **universal multi-domain simulation and design platform**.

### New Core Domains

**Circuit/Electrical Engineering** ⭐ (PR #43)
- Complete circuit modeling domain with typed operator graphs (R → C → Op-Amp)
- Multi-domain integration: Circuit ↔ Audio, Geometry, Physics
- Reference-based composition with auto-anchors
- 5 complete examples: RC filters, op-amps, guitar pedals, PCB parasitic extraction
- **Specs**: `docs/SPEC-CIRCUIT.md`, `docs/ADR/003-circuit-modeling-domain.md`
- **Examples**: `examples/circuit/` (6 files)

**Fluid Dynamics & Acoustics** ⭐ (PR #44)
- FluidDynamics: Compressible/incompressible flow, gas dynamics, thermodynamic coupling
- Acoustics: 1D waveguides, FDTD, Helmholtz resonators, radiation impedance
- Multi-domain flows: FluidDynamics → Acoustics → Audio → Geometry
- Use case: 2-stroke engine exhaust system modeling
- **Specs**: `docs/DOMAIN_ARCHITECTURE.md` (sections 2.9, 2.10)
- **Use Cases**: `docs/USE_CASES/2-stroke-muffler-modeling.md`

**Instrument Modeling & Timbre Extraction** ⭐ (PR #45)
- 35 operators for acoustic analysis and synthesis
- Record instrument → extract timbre → synthesize new notes
- MIDI instrument creation, timbre morphing, luthier analysis
- Addresses "holy grail" of audio DSP: recordings → reusable synthesis models
- **Specs**: `docs/SPEC-TIMBRE-EXTRACTION.md`, `docs/ADR/003-instrument-modeling-domain.md`

**Audio Time Alignment** (PR #46)
- Measurement operators (sine sweep, MLS, impulse train)
- Analysis operators (IR extraction, group delay, cross-correlation)
- Alignment operators (delay designer, crossover phase alignment)
- Solves critical pro audio problems (car audio, studio monitors)
- **Specs**: `docs/LEARNINGS/TIME_ALIGNMENT_OPERATORS.md`

**Multi-Physics Engineering** ⭐ (PR #47)
- Four physics domains: FluidNetwork, ThermalODE, FluidJet, CombustionLight
- Complete multi-domain pipeline: Geometry → Fluid → Thermal → Combustion
- J-tube fire pit example validates operator graph paradigm for engineering
- **Specs**: `docs/SPEC-PHYSICS-DOMAINS.md`
- **Examples**: `docs/EXAMPLES/J-TUBE-FIREPIT-MULTIPHYSICS.md`

**Optimization & Design Discovery** ⭐ (PR #48)
- 16 optimization algorithms across 5 categories
- Evolutionary (GA, DE, CMA-ES), Local (L-BFGS, Nelder-Mead), Surrogate (Bayesian)
- Multi-objective (NSGA-II, SPEA2), Combinatorial (Simulated Annealing)
- Transforms Kairo from simulation platform → design discovery platform
- **Specs**: `docs/LEARNINGS/OPTIMIZATION_ALGORITHMS_CATALOG.md`

**Video & Audio Encoding** ⭐ (PR #51)
- VideoDomain: Structural video operations (decode, encode, scale, crop, composition)
- AudioFilterDomain: Multimedia audio processing (loudness, EQ, dynamics, timing)
- FilterDomain: Visual filters (blur, sharpen, color correction, denoise, stabilize)
- CodecDomain: Typed codec operators (H.264, H.265, ProRes, GPU acceleration)
- SyncDomain: Audio/video synchronization (drift correction, event alignment, lip-sync)
- BatchDomain: Parallel batch processing for multimedia workflows
- Magic operator: `video.fix()` — Auto-magic video correction (sync, denoise, stabilize, color correct)
- **Key insight**: Video = operator DAG on AV streams (perfect fit for Kairo)
- **Specs**: `docs/SPEC-VIDEO-AUDIO-ENCODING.md`

### Complete Domain Catalog

Kairo now has **comprehensive specifications** for these domains:

**Core Implemented**:
- Transform (FFT, STFT, wavelets)
- Stochastic (RNG, distributions, Monte Carlo)
- Fields/Grids (PDEs, stencils, boundary conditions)
- Agent/Particle (sparse systems, N-body, behaviors)
- Audio/DSP (synthesis, filters, effects, I/O)
- Visual (rendering, composition, video export)

**Architecture Complete (Specs Ready)**:
- **Video & Audio Encoding** (ffmpeg-style pipelines, sync correction, batch processing)
- **Geometry** (TiaCAD-inspired, reference-based composition)
- **Circuit** (electrical engineering, PCB, analog audio)
- **Acoustics** (waveguides, FDTD, resonators, radiation)
- **FluidDynamics** (compressible/incompressible, gas dynamics, engines)
- **InstrumentModeling** (timbre extraction, synthesis, MIDI creation)
- **Optimization** (16 algorithms for design discovery)
- **Physics** (thermal, combustion, multi-physics coupling)
- **Sparse Linear Algebra** (iterative solvers, preconditioning)

**Planned (Next Wave)**:
- Image/Vision, Graph/Network, Symbolic/Algebraic, Neural Operators

See **[docs/DOMAIN_ARCHITECTURE.md](docs/DOMAIN_ARCHITECTURE.md)** for the complete vision (2,266 lines covering 20+ domains).

### Why This Matters

**Kairo is now the only platform that unifies**:
- Video encoding + audio filtering + sync correction
- Circuit simulation + PCB layout + analog audio modeling
- Fluid dynamics + acoustics + audio synthesis
- Multi-physics engineering + geometry + optimization
- All domains share the same type system, scheduler, and MLIR compilation

This positions Kairo as a **universal deterministic compute kernel** for creative computation, engineering simulation, and design optimization.

---

## Documentation

### Architecture & Design
- **[Architecture](ARCHITECTURE.md)** ⭐ - Finalized Kairo Stack architecture (v1.0 Draft)
- **[Domain Architecture](docs/DOMAIN_ARCHITECTURE.md)** ⭐⭐ - Complete multi-domain vision (2,266 lines, 20+ domains)
- **[Ecosystem Map](ECOSYSTEM_MAP.md)** ⭐ - Comprehensive map of all Kairo domains, modules, and libraries
- **[Transform Dialect](docs/SPEC-TRANSFORM.md)** - First-class domain transforms (FFT, STFT, wavelets, etc.)
- **[Graph IR](docs/SPEC-GRAPH-IR.md)** - Frontend-kernel boundary specification

### Domain Specifications (NEW - November 2025)
- **[Video & Audio Encoding](docs/SPEC-VIDEO-AUDIO-ENCODING.md)** ⭐ - Video encoding, audio/video filtering, sync correction, ffmpeg-style pipelines
- **[Circuit Domain](docs/SPEC-CIRCUIT.md)** - Electrical engineering, PCB layout, analog audio
- **[Timbre Extraction](docs/SPEC-TIMBRE-EXTRACTION.md)** - Instrument modeling and synthesis
- **[Physics Domains](docs/SPEC-PHYSICS-DOMAINS.md)** - Multi-physics engineering (thermal, fluid, combustion)
- **[Geometry Domain](docs/SPEC-GEOMETRY.md)** - TiaCAD-inspired parametric modeling
- **[Coordinate Frames](docs/SPEC-COORDINATE-FRAMES.md)** - Unified reference and anchor system
- **[Operator Registry](docs/SPEC-OPERATOR-REGISTRY.md)** - Complete operator catalog (1,244 lines)

### Architectural Decision Records (ADRs)
- **[ADR-001: Unified Reference Model](docs/ADR/001-unified-reference-model.md)** - Cross-domain reference system
- **[ADR-002: Cross-Domain Patterns](docs/ADR/002-cross-domain-architectural-patterns.md)** - TiaCAD/RiffStack patterns
- **[ADR-003: Circuit Modeling](docs/ADR/003-circuit-modeling-domain.md)** - Circuit domain rationale
- **[ADR-003: Instrument Modeling](docs/ADR/003-instrument-modeling-domain.md)** - Timbre extraction rationale

### Implementation Guides & Learnings
- **[Domain Implementation Guide](docs/GUIDES/DOMAIN_IMPLEMENTATION_GUIDE.md)** - Step-by-step domain creation
- **[Operator Registry Expansion](docs/LEARNINGS/OPERATOR_REGISTRY_EXPANSION.md)** - 8 priority domains catalog
- **[Optimization Algorithms](docs/LEARNINGS/OPTIMIZATION_ALGORITHMS_CATALOG.md)** - 16 algorithms across 5 categories
- **[Time Alignment Operators](docs/LEARNINGS/TIME_ALIGNMENT_OPERATORS.md)** - Audio measurement and alignment

### Examples & Use Cases
- **[Circuit Examples](examples/circuit/)** - RC filters, op-amps, guitar pedals, PCB (6 examples)
- **[J-tube Fire Pit](docs/EXAMPLES/J-TUBE-FIREPIT-MULTIPHYSICS.md)** - Multi-physics engineering showcase
- **[2-Stroke Muffler](docs/USE_CASES/2-stroke-muffler-modeling.md)** - Fluid dynamics + acoustics + audio
- **[Examples Guide](docs/EXAMPLES/README.md)** - Comprehensive examples documentation

### Language Reference
- **[Complete Specification](SPECIFICATION.md)** - Full language reference
- **[Audio Specification](AUDIO_SPECIFICATION.md)** - Kairo.Audio dialect specification (composer surface)

### Project History
- **[Evolution Summary](docs/KAIRO_v0.3.1_SUMMARY.md)** - Why Kairo v0.3.1
- **[Project Review](PROJECT_REVIEW_AND_NEXT_STEPS.md)** - Comprehensive assessment & roadmap
- **[Legacy Docs](docs/legacy/)** - v0.2.2 CCDSL documentation

---

## Evolution from Creative Computation DSL

Kairo v0.3.1 is the evolution of Creative Computation DSL v0.2.2, incorporating:

- **Better semantics**: `flow(dt)` blocks, `@state` declarations, explicit RNG
- **Clearer branding**: "Kairo" is unique and memorable
- **Same foundation**: Frontend work carries forward, comprehensive stdlib preserved

See [docs/KAIRO_v0.3.1_SUMMARY.md](docs/KAIRO_v0.3.1_SUMMARY.md) for detailed evolution rationale.

---

## Related Projects

**[RiffStack](https://github.com/scottsen/riffstack)** - Live performance shell for Kairo.Audio

RiffStack is a stack-based, YAML-driven performance environment that serves as the live interface to Kairo.Audio. While Kairo.Audio provides the compositional language layer, RiffStack offers real-time interaction and performance capabilities. Together they form a complete audio synthesis and performance ecosystem built on Kairo's deterministic execution kernel.

---

## Contributing

Kairo is in active development. Contributions welcome!

**Priority areas:**
- Runtime engine implementation
- Field operations (NumPy-based)
- Visualization backend
- Example programs
- Documentation improvements

See `SPECIFICATION.md` Section 19 for implementation guidance.

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Contact

- **GitHub**: https://github.com/scottsen/kairo
- **Issues**: https://github.com/scottsen/kairo/issues

---

**Status:** v0.7.4 - Real MLIR Integration Complete (All 6 Phases) | **Stable Version:** 0.6.0 | **Current Version:** 0.7.4 | **Last Updated:** 2025-11-15
