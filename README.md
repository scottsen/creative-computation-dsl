---
project: kairo
type: software
status: active
beth_topics:
- kairo
- creative-computation
- dsl
- mlir
tags:
- compiler
- simulation
- generative
- deterministic
---

# Kairo v0.6.0

**A semantic, deterministic transform kernel with two human-friendly faces:**
- **Kairo.Audio** for composition
- **RiffStack** for performance

*Where computation becomes composition*

> 📐 **Architecture**: See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete Kairo Stack design (kernel, frontends, Graph IR, transforms)

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

**Version**: 0.6.0
**Status**: Alpha - Core Features + Agent + Audio + I/O Complete

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

### 🚧 Experimental (Text-Based, Not Production)
- **MLIR text IR generation** (not real MLIR bindings)
- Optimization passes (basic constant folding, DCE stubs)

### 📋 Planned (Not Yet Implemented)
- **Real MLIR Integration** - Currently text-based IR only
- **Native Code Generation** - Requires real MLIR bindings
- **Physical Unit Checking** - Annotations exist, not enforced
- **Hot-reload** - Architecture designed, not implemented

**Next Milestone**: v0.7.0 Real MLIR Integration (12+ months)

---

## Documentation

### Architecture & Design
- **[Architecture](ARCHITECTURE.md)** ⭐ - Finalized Kairo Stack architecture (v1.0 Draft)
- **[Ecosystem Map](ECOSYSTEM_MAP.md)** ⭐ - Comprehensive map of all Kairo domains, modules, and libraries
- **[Transform Dialect](docs/SPEC-TRANSFORM.md)** - First-class domain transforms (FFT, STFT, wavelets, etc.)
- **[Graph IR](docs/SPEC-GRAPH-IR.md)** - Frontend-kernel boundary specification
- **[Architecture Analysis](ARCHITECTURE_ANALYSIS.md)** - Historical architectural review

### Language Reference
- **[Complete Specification](SPECIFICATION.md)** - Full language reference
- **[Audio Specification](AUDIO_SPECIFICATION.md)** - Kairo.Audio dialect specification (composer surface)
- **[MLIR Pipeline Status](MLIR_PIPELINE_STATUS.md)** - Complete compilation pipeline details

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

**Status:** Alpha - Core Features + Agent + Audio + I/O Complete | **Version:** 0.6.0 | **Last Updated:** 2025-11-14
