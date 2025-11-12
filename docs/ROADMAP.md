# Kairo Semantic Kernel Roadmap

This roadmap outlines the evolution of Kairo from a domain-specific language into a **semantic kernel** that provides deterministic, multi-domain computation infrastructure for creative and scientific DSLs.

---

## Milestone 1 — Kernel Types & Scheduler

**Goal**: Implement the core semantic kernel types and multirate scheduler.

**Status**: 🚧 Planning

### Tasks

- [ ] **Stream Supertype**
  - [ ] Implement `Stream<T,Domain,Rate>` as unified type
  - [ ] Maintain backward-compatible aliases (`Signal<T>`, `FieldND<T>`)
  - [ ] Add domain/rate metadata to IR
  - [ ] Update type checker to handle parameterized streams

- [ ] **Event Streams**
  - [ ] Implement `Evt<A>` as first-class type
  - [ ] `event.from_list([...])` constructor
  - [ ] `event.map(fn)`, `event.merge(...)` operators
  - [ ] `event.to_signal(mode=hold|impulse)` conversion with sample-accurate timing

- [ ] **Multirate Scheduler Core**
  - [ ] Rate declarations: `@rate=audio|control|visual|sim`
  - [ ] LCM partition computation
  - [ ] Topological execution within partitions
  - [ ] Sample-accurate event fence insertion

- [ ] **Resampling**
  - [ ] `resample(stream, to=rate, mode=nearest|linear|cubic)`
  - [ ] Deterministic rounding modes
  - [ ] Test vectors for cross-platform conformance

- [ ] **Profile System**
  - [ ] Profile declaration syntax
  - [ ] Precedence enforcement (per-op > solver > module > global)
  - [ ] Determinism level checking (`bitexact|repro|replayable|best_effort`)
  - [ ] Precision selection (f64/f32/f16)

- [ ] **Boundary/Interface Metadata**
  - [ ] `Boundary` type and syntax (no solver changes yet)
  - [ ] `Interface` type and syntax
  - [ ] Attachment to streams/spaces
  - [ ] IR representation

- [ ] **Introspection CLI**
  - [ ] `kairo graph --json` command
  - [ ] JSON schema for graph output (nodes, edges, rates, profiles)
  - [ ] Basic visualizer for graphs

**Deliverables**:
- Kernel types implemented in compiler
- Multirate scheduler working for simple examples
- Profile system enforcing determinism levels
- CLI introspection tool

**Target**: 6-8 weeks

---

## Milestone 2 — Backends & Contracts

**Goal**: Bind to real MLIR backends and enforce boundary/interface contracts in solvers.

**Status**: 📋 Planned

### Tasks

- [ ] **MLIR Backend Integration**
  - [ ] Emit to `linalg` dialect (dense field ops)
  - [ ] Emit to `affine` dialect (loop optimization)
  - [ ] Emit to `vector` dialect (SIMD)
  - [ ] Emit to `gpu` dialect (GPU kernels)
  - [ ] Emit to `async` dialect (async I/O, audio sinks)

- [ ] **CPU Pipeline**
  - [ ] Lower to LLVM dialect
  - [ ] LLVM optimization passes
  - [ ] Native code generation (x86-64, ARM)
  - [ ] Vectorization (AVX2, NEON)

- [ ] **GPU Pipeline**
  - [ ] Lower to SPIR-V (Vulkan/OpenCL)
  - [ ] Lower to Metal (macOS/iOS)
  - [ ] Lower to CUDA (NVIDIA)
  - [ ] GPU memory management (deterministic allocation)

- [ ] **Audio Device Sink**
  - [ ] Async output stream via `async` dialect
  - [ ] Deterministic block size (e.g., 512 samples)
  - [ ] Cross-platform audio backend (PortAudio, JACK, CoreAudio)
  - [ ] Low-latency mode (live profile)

- [ ] **BoundaryContract Enforcement**
  - [ ] Solvers declare contracts (`apply=pre|post, mode=ghost|mirror`)
  - [ ] Compiler inserts boundary application at correct point
  - [ ] Ghost cell generation (deterministic halo exchange)
  - [ ] Test against existing PDE examples

- [ ] **Interface Coupling**
  - [ ] Grid interpolation for cross-domain interfaces
  - [ ] Flux matching implementation
  - [ ] Multi-domain solver support (coupled diffusion, etc.)

**Deliverables**:
- Real MLIR backend code generation
- CPU and GPU pipelines working
- Audio output functional
- Boundary/interface contracts enforced

**Target**: 8-10 weeks

---

## Milestone 3 — Dialects & Examples

**Goal**: Implement domain dialects (starting with Kairo.Audio/StreamTone) and upgrade examples to use kernel features.

**Status**: 📋 Planned

### Tasks

- [ ] **Kairo.Audio (StreamTone) — Phase 1**
  - [ ] Oscillators: `sine`, `saw`, `square`, `tri`, `noise`
  - [ ] Filters: `lpf`, `hpf`, `bpf`
  - [ ] Envelopes: `adsr`, `ar`, `envexp`
  - [ ] Utilities: `mix`, `pan`, `clip`, `db2lin`
  - [ ] Event-to-signal conversion
  - [ ] Audio-rate examples

- [ ] **Kairo.Audio — Phase 2**
  - [ ] Effects: `delay`, `reverb`, `chorus`, `flanger`
  - [ ] Convolution (FFT backend)
  - [ ] Waveguide string (physical modeling)
  - [ ] Body/pickup/amp/cab models

- [ ] **Example Upgrades**
  - [ ] PDE examples with `Boundary` objects
  - [ ] Multi-domain PDE with `Interface` coupling
  - [ ] Audio synthesis examples (FM, subtractive)
  - [ ] Physical modeling examples (plucked string)
  - [ ] Audio-visual coupling (spectrum → particles)

- [ ] **Golden Artifact Tests**
  - [ ] WAV file output tests (bitexact audio)
  - [ ] PNG image output tests (visual examples)
  - [ ] CI integration (regression detection)

**Deliverables**:
- StreamTone audio library functional
- Upgraded examples showcasing kernel features
- Golden artifact tests in CI

**Target**: 6-8 weeks

---

## Milestone 4 — Tooling & Docs

**Goal**: Polish tooling, introspection, visualization, and documentation for public release.

**Status**: 📋 Planned

### Tasks

- [ ] **Introspection**
  - [ ] JSON schema finalized and documented
  - [ ] Graph introspection with full metadata
  - [ ] Profile introspection API
  - [ ] Rate/domain query utilities

- [ ] **Visualization**
  - [ ] `visualize(graph)` — dataflow graph rendering
  - [ ] `visualize(boundary, field=...)` — boundary condition overlay
  - [ ] `visualize(interface, fields=...)` — interface coupling display
  - [ ] `visualize(signal)` — waveform/spectrum display

- [ ] **Documentation**
  - [ ] Full SPEC refresh (incorporate all RFCs)
  - [ ] Tutorial series (getting started → advanced)
  - [ ] Audio synthesis guide
  - [ ] PDE solver guide
  - [ ] Multi-domain coupling guide
  - [ ] Website/docs site (e.g., MkDocs, Docusaurus)

- [ ] **Developer Tools**
  - [ ] `kairo check` — type/determinism checking without execution
  - [ ] `kairo profile` — performance profiling
  - [ ] `kairo test` — test runner with golden artifacts
  - [ ] Language server protocol (LSP) for IDE support

**Deliverables**:
- Complete documentation suite
- Introspection/visualization tools
- Developer-friendly CLI
- LSP for editor integration

**Target**: 6 weeks

---

## Milestone 5 — Advanced Dialects & Research

**Goal**: Extend to additional domain dialects and research features.

**Status**: 📋 Future

### Potential Dialects

- **Luma** (visual/image processing)
  - Image kernels, compositing, color spaces
  - Real-time shader integration
  - Video output

- **Asterion** (astrophysics)
  - N-body simulation
  - Hydrodynamics (SPH, grid-based)
  - Radiative transfer

- **Bio** (biological simulation)
  - Reaction-diffusion on manifolds
  - Agent-based epidemiology
  - Cellular automata

### Research Features

- [ ] **Adaptive Timesteps** (solver-controlled `dt`)
- [ ] **Mesh Support** (unstructured grids)
- [ ] **Distributed Execution** (MPI-based multi-node)
- [ ] **Probabilistic Programming** (Bayesian inference DSL)
- [ ] **Differentiable Programming** (autodiff for optimization)

**Target**: Ongoing

---

## Issue Tracking

Track progress via GitHub issues with labels:

- `milestone:1-kernel` — Milestone 1 tasks
- `milestone:2-backends` — Milestone 2 tasks
- `milestone:3-dialects` — Milestone 3 tasks
- `milestone:4-tooling` — Milestone 4 tasks
- `milestone:5-research` — Milestone 5 tasks

Additional labels:
- `rfc` — RFC/design discussions
- `breaking-change` — Potential API breakage
- `good-first-issue` — Newcomer-friendly tasks
- `performance` — Optimization work
- `documentation` — Docs improvements

---

## Timeline Summary

| Milestone | Duration | Cumulative | Target Completion |
|-----------|----------|------------|-------------------|
| M1: Kernel Types & Scheduler | 6-8 weeks | 6-8 weeks | Q2 2025 |
| M2: Backends & Contracts | 8-10 weeks | 14-18 weeks | Q3 2025 |
| M3: Dialects & Examples | 6-8 weeks | 20-26 weeks | Q4 2025 |
| M4: Tooling & Docs | 6 weeks | 26-32 weeks | Q1 2026 |
| M5: Advanced Dialects | Ongoing | — | 2026+ |

**Public v1.0 Release Target**: Q1 2026

---

## Success Metrics

- ✅ **Determinism**: All examples bitexact-reproducible (with `strict` profile)
- ✅ **Performance**: GPU examples ≥10× faster than NumPy baseline
- ✅ **Usability**: StreamTone audio examples competitive with SuperCollider/CSound
- ✅ **Interop**: Seamless audio+visual+sim examples
- ✅ **Adoption**: 100+ stars on GitHub, community contributions

---

## Open Questions & RFCs

1. **Exact JSON schema for introspection** (Milestone 1)
2. **Canonical resampling modes & rounding** (Milestone 1)
3. **Third-party solver determinism contracts** (Milestone 2)
4. **Curved/moving boundaries** (Milestone 3+)
5. **Polyphony & voice management** (Milestone 3+)
6. **Adaptive timesteps** (Milestone 5)
7. **Distributed execution model** (Milestone 5)

Track these as RFC issues: `rfc:introspection-schema`, `rfc:resampling`, etc.

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute to the roadmap.

**Priority areas**:
- Kernel type implementation
- MLIR backend expertise
- Audio DSP library contributions
- PDE solver integration
- Documentation improvements

---

**Last Updated**: 2025-11-12
