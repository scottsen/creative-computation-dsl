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

# Kairo

> *Where computation becomes composition*

**Kairo** is a universal, deterministic computation platform that unifies domains that have never talked to each other before: **audio synthesis meets physics simulation meets circuit design meets geometry meets optimization** — all in one type system, one scheduler, one language.

## Why Kairo Exists

Current tools force you to:
- Export CAD → import to FEA → export mesh → import to CFD → manually couple results
- Write audio DSP in C++ → physics in Python → visualization in JavaScript
- Bridge domains with brittle scripts and incompatible data formats

**Kairo eliminates this fragmentation.** Model a guitar string's physics, synthesize its sound, optimize its geometry, and visualize the result — all in the same deterministic execution environment.

## Two Surfaces, One Kernel

Kairo presents **two human-friendly faces** powered by a single semantic kernel:

- **Kairo.Audio** — Declarative language for compositional audio, physics, and multi-domain scenes
- **RiffStack** — Live performance environment for real-time interaction and improvisation

Both compile to the same Graph IR, share the same operator registry, and guarantee deterministic, reproducible results.

> 📐 **Deep Dive**: See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete stack design (kernel, frontends, Graph IR, MLIR compilation)

## What Makes Kairo Different

**Cross-Domain Composition**
- Audio synthesis + fluid dynamics + circuit simulation in the same program
- Type-safe connections between domains (e.g., field → agent force, geometry → audio impulse response)
- Single execution model handles multiple rates (audio @ 48kHz, control @ 60Hz, physics @ 240Hz)

**Deterministic by Design**
- Bitwise-identical results across runs, platforms, and GPU vendors
- Explicit RNG seeding, sample-accurate event scheduling
- Three profiles: `strict` (bit-exact), `repro` (deterministic FP), `live` (low-latency)

**Transform-First Thinking**
- FFT, STFT, wavelets, DCT as first-class operations
- Domain changes (time ↔ frequency, space ↔ k-space) are core primitives
- Uniform transform API across all domains

**Production-Grade Compilation**
- MLIR-based compiler with 6 custom dialects
- Lowers to optimized CPU/GPU code via LLVM
- Field operations, agents, audio DSP, temporal execution all compile to native code

---

## Cross-Domain in Action

Here's what sets Kairo apart — domains working together seamlessly:

```kairo
# Couple fluid dynamics → acoustics → audio synthesis
use fluid, acoustics, audio

# Simulate airflow in a 2-stroke engine exhaust
@state flow : FluidNetwork1D = engine_exhaust(length=2.5m, diameter=50mm)
@state acoustic : AcousticField1D = waveguide_from_flow(flow)

flow(dt=0.1ms) {
    # Fluid dynamics: pressure pulses from engine
    flow = flow.advance(engine_pulse(t), method="lax_wendroff")

    # Couple to acoustics: flow → sound propagation
    acoustic = acoustic.couple_from_fluid(flow, impedance_match=true)

    # Synthesize audio from acoustic field
    let exhaust_sound = acoustic.to_audio(mic_position=1.5m)

    # Real-time output
    audio.play(exhaust_sound)
}
```

**One program. Three domains. Zero glue code.**

See [docs/use-cases/2-stroke-muffler-modeling.md](docs/use-cases/2-stroke-muffler-modeling.md) for the complete example.

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

## The Ecosystem Vision

Kairo's domain architecture has been massively expanded in November 2025, establishing it as a **universal multi-domain platform**.

### Domain Coverage (20+ Domains Specified)

**Production-Ready** (v0.6-0.7):
- Audio/DSP, Fields/Grids, Agents/Particles, Visual Rendering, Transform Dialect

**Architecture Complete** (Comprehensive Specs):
- Circuit Design & Analog Electronics
- Fluid Dynamics & Acoustics
- Instrument Modeling & Timbre Extraction
- Video/Audio Encoding & Synchronization
- Multi-Physics Engineering (Thermal, Combustion, FluidJet)
- Optimization (16 algorithms: GA, CMA-ES, Bayesian, NSGA-II)
- Geometry & Parametric CAD
- Chemistry & Molecular Dynamics
- Procedural Generation & Emergence

**Planned**:
- Graph/Network Analysis, Symbolic Math, Neural Operators, BI/Analytics

**Why This Matters**: These aren't isolated silos — they're integrated domains sharing:
- One type system (with physical units)
- One scheduler (multirate, sample-accurate)
- One compiler (MLIR → LLVM/GPU)
- One determinism model (strict/repro/live profiles)

**Circuit simulation can drive audio synthesis. Fluid dynamics can generate acoustic fields. Geometry can define boundary conditions for PDEs. Optimization can tune parameters across all domains.**

> 📚 **Complete Vision**: See [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md) for the full ecosystem architecture and [docs/architecture/domain-architecture.md](docs/architecture/domain-architecture.md) for deep technical specifications (2,266 lines covering all domains)

---

## Professional Applications & Long-Term Vision

Kairo's unified multi-domain architecture addresses fundamental problems across professional fields:

### Engineering & Design
**Current Pain**: CAD → FEA → CFD → optimization requires manual data export/import, incompatible formats, weeks of iteration
**Kairo Solution**: Geometry → Physics → Optimization in one deterministic pipeline
- **Aerospace**: Optimize wing geometry based on CFD results and structural constraints
- **Automotive**: Couple exhaust acoustics + fluid dynamics + thermal analysis for muffler design
- **HVAC**: Model airflow + thermal + acoustic performance in building ventilation systems

### Audio Production & Lutherie
**Current Pain**: Physical modeling requires separate tools for mechanics, acoustics, and DSP
**Kairo Solution**: Physics → Acoustics → Audio synthesis in unified framework
- Record acoustic guitar → extract timbre → create playable virtual instrument
- Design guitar body geometry → simulate acoustics → hear the sound before building
- Model pickup placement + circuit design → optimize tone before winding coils

### Scientific Computing
**Current Pain**: Multi-physics simulations require coupling incompatible solvers (COMSOL + MATLAB + custom code)
**Kairo Solution**: Unified PDE solver + Monte Carlo + optimization + visualization
- Chemistry: Molecular dynamics + reaction kinetics + thermodynamics
- Ecology: Agent-based modeling + field diffusion + spatial statistics
- Climate: Fluid dynamics + thermal transport + stochastic processes

### Creative Coding & Generative Art
**Current Pain**: Real-time graphics + procedural audio + physics simulation = three separate frameworks
**Kairo Solution**: All creative domains in one deterministic, reproducible environment
- Couple particle systems to audio synthesis (visual state → sound parameters)
- Procedural geometry generation driven by audio analysis
- Deterministic generative art: same seed = identical output every time

### Finance & Risk Analysis
**Current Pain**: Monte Carlo in R + optimization in Python + visualization in Tableau = integration hell
**Kairo Solution**: Stochastic processes + optimization + real-time analytics
- GPU-accelerated Monte Carlo with 100x speedup
- Optimize portfolio with multi-objective algorithms (NSGA-II, CMA-ES)
- Deterministic backtesting: reproducible risk analysis

**Key Insight**: These fields don't need *separate tools* — they need *integrated domains*. Kairo is the only platform that unifies them with a single type system, scheduler, and compiler.

> 💼 **Professional Impact Analysis**: See [docs/reference/professional-domains.md](docs/reference/professional-domains.md) for detailed value proposition across 10+ professional fields

---

## Documentation

> 📚 **Start Here**: [docs/README.md](docs/README.md) — Complete documentation navigation guide

### Essential Reading

**Architecture & Vision**
- **[Architecture](ARCHITECTURE.md)** ⭐ — The Kairo Stack: kernel, frontends, Graph IR, MLIR compilation
- **[Ecosystem Map](ECOSYSTEM_MAP.md)** ⭐ — Complete map of all domains, modules, and expansion roadmap
- **[Domain Architecture](docs/architecture/domain-architecture.md)** — Deep technical vision (2,266 lines, 20+ domains)

**Getting Started**
- **[Getting Started Guide](docs/getting-started.md)** — Installation, first program, core concepts
- **[Language Specification](SPECIFICATION.md)** — Complete Kairo language reference
- **[Audio Specification](AUDIO_SPECIFICATION.md)** — Kairo.Audio compositional DSL

**Professional Applications**
- **[Professional Domains](docs/reference/professional-domains.md)** ⭐ — Value proposition across engineering, science, finance, creative fields
- **[Use Cases](docs/use-cases/)** — Real-world applications (2-stroke muffler, chemistry framework)
- **[Examples](docs/examples/)** — Working examples (multi-physics, emergence, cross-domain)

### Domain Specifications (19 Comprehensive Specs)

All specifications are in **[docs/specifications/](docs/specifications/)**:
- **Circuit**, **Chemistry**, **Emergence**, **Procedural Generation**, **Video/Audio Encoding**
- **Geometry**, **Coordinate Frames**, **Physics Domains**, **Timbre Extraction**
- **Graph IR**, **MLIR Dialects**, **Operator Registry**, **Scheduler**, **Transform**, **Type System**
- **Profiles**, **Snapshot ABI**, **BI Domain**, **KAX Language**

See [docs/specifications/README.md](docs/specifications/README.md) for full catalog.

### Architectural Decision Records

See **[docs/adr/](docs/adr/)** for why key decisions were made:
- Unified Reference Model, Cross-Domain Patterns, Circuit/Instrument/Chemistry Domains, GPU-First Approach

### Implementation Resources

- **[Domain Implementation Guide](docs/guides/domain-implementation.md)** — How to add new domains
- **[Reference Catalogs](docs/reference/)** — Operator catalogs, patterns, domain overviews
- **[Roadmap](docs/roadmap/)** — MVP, v0.1, implementation progress, testing strategy

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

Kairo is building toward something transformative: a universal platform where professional domains that have never talked before can seamlessly compose. Contributions welcome at all levels!

### High-Impact Areas

**Domain Expansion** — Help implement new domains:
- Geometry/CAD integration (TiaCAD-inspired reference system)
- Chemistry & molecular dynamics
- Graph/network analysis
- Neural operator support

**Core Infrastructure** — Strengthen the foundation:
- MLIR lowering passes and optimization
- GPU acceleration for field operations
- Multi-GPU support and distributed execution
- Cross-domain type checking and unit validation

**Professional Applications** — Build real-world examples:
- Engineering workflows (CAD → FEA → optimization)
- Scientific computing (multi-physics simulations)
- Audio production (lutherie, timbre extraction)
- Creative coding (generative art, live visuals)

**Documentation & Education**
- Tutorials for specific domains
- Professional field guides
- Implementation examples
- Performance benchmarks

### Getting Involved

1. **Explore** — Read [ARCHITECTURE.md](ARCHITECTURE.md) and [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md)
2. **Pick a Domain** — See [docs/architecture/domain-architecture.md](docs/architecture/domain-architecture.md) for specs
3. **Follow the Guide** — Use [docs/guides/domain-implementation.md](docs/guides/domain-implementation.md)
4. **Join the Vision** — Help build the future of multi-domain computation

See [SPECIFICATION.md](SPECIFICATION.md) Section 19 for detailed implementation guidance.

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Contact

- **GitHub**: https://github.com/scottsen/kairo
- **Issues**: https://github.com/scottsen/kairo/issues

---

**Status:** v0.7.4 - Real MLIR Integration Complete (All 6 Phases) | **Stable Version:** 0.6.0 | **Current Version:** 0.7.4 | **Last Updated:** 2025-11-15
