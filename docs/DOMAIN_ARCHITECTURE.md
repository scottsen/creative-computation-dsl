# Kairo Domain Architecture

**Version:** 1.0
**Status:** Vision Document
**Last Updated:** 2025-11-15

---

## Overview

This document presents a comprehensive, forward-looking view of the domains and layers Kairo will eventually encompass. These domains emerge naturally from building a **deterministic, multi-domain semantic compute kernel** designed for audio, physics, graphics, AI, simulation, and analytics.

This is not aspirational fluff — these are the domains that consistently appear in successful multi-modal compute systems. Each domain is justified by real computational needs and integrated into Kairo's unified type system, scheduler, and MLIR compilation pipeline.

### Document Purpose

- **Current Reference**: Understand what domains exist today
- **Planning Guide**: Inform roadmap prioritization
- **Architecture Vision**: Ensure coherent integration across domains
- **Engineering Resource**: Define operator requirements and dependencies

### Related Documentation

This document is part of a comprehensive domain architecture learning system:

- **[ADR-002: Cross-Domain Architectural Patterns](ADR/002-cross-domain-architectural-patterns.md)** — Battle-tested patterns from TiaCAD, RiffStack, and Strudel (reference systems, auto-anchors, operator registries, passes)
- **[Domain Implementation Guide](GUIDES/DOMAIN_IMPLEMENTATION_GUIDE.md)** — Step-by-step guide for implementing new domains (checklists, templates, best practices)
- **[Operator Registry Expansion](LEARNINGS/OPERATOR_REGISTRY_EXPANSION.md)** — Detailed catalog of 7 priority domains with complete operator specifications (Audio, Physics, Geometry, Finance, Graphics, Neural, Pattern)

**For domain implementers**: Start with ADR-002 for architectural principles, then follow the Domain Implementation Guide for practical steps.

---

## Domain Classification

Domains are organized into three tiers based on urgency and system maturity:

1. **Core Domains** — Essential for audio, fields, physics, graphics, or simulation. Must have.
2. **Next-Wave Domains** — Naturally emerge from a multirate, GPU/CPU-pluggable, graph-IR-based kernel. Highly likely.
3. **Advanced Domains** — Future expansion for specialized use cases. May add later.

---

## 1. Core Domains (MUST HAVE)

These domains form the bare minimum for a universal transform/simulation kernel. Several are already partially defined in `SPEC-MLIR-DIALECTS.md` and operational in v0.7.0.

---

### 1.1 Transform Dialect

**Purpose**: Domain transforms between time/frequency, space/k-space, and other spectral representations.

**Why Essential**: Audio processing, signal analysis, PDE solving, and compression all require fast, accurate transforms.

**Status**: ✅ Partially implemented (FFT, STFT, IFFT in kairo.transform dialect)

**Operators**:
- `fft` / `ifft` — Fast Fourier Transform (1D)
- `fft2d` / `ifft2d` — 2D FFT (space → k-space)
- `stft` / `istft` — Short-Time Fourier Transform
- `dct` / `idct` — Discrete Cosine Transform
- `wavelet` — Wavelet transforms (Haar, Daubechies, etc.)
- `mel` — Mel-frequency transforms
- `cepstral` — Cepstral analysis
- `reparam` — Reparameterization (e.g., exponential → linear frequency)

**Dependencies**: Linear algebra, windowing functions

**References**: `SPEC-TRANSFORM.md`, `SPEC-MLIR-DIALECTS.md`

---

### 1.2 Stochastic Dialect

**Purpose**: Random number generation, distributions, stochastic processes, Monte Carlo simulation.

**Why Essential**: Agent mutation, noise generation, probabilistic simulation, and procedural content all require deterministic, high-quality randomness.

**Status**: ⚙️ In progress (Philox RNG implemented, distribution ops planned)

**Operators**:
- `rng.init` — Initialize RNG state with seed
- `rng.uniform` — Uniform distribution [0, 1)
- `rng.normal` — Gaussian distribution (mean, stddev)
- `rng.exponential` — Exponential distribution (rate)
- `rng.poisson` — Poisson distribution (lambda)
- `monte_carlo.integrate` — Monte Carlo integration
- `sde.step` — Stochastic differential equation step (Euler-Maruyama, Milstein)

**Dependencies**: None (foundational)

**Determinism**: Strict (Philox 4×32-10 with hash-based seeding)

---

### 1.3 Fields / Grids Dialect

**Purpose**: Operations on scalar/vector/tensor fields, stencils, PDE operators, boundary conditions.

**Why Essential**: Fluid simulation, reaction-diffusion, heat transfer, and electromagnetic fields all operate on spatial grids.

**Status**: ✅ Partially implemented (kairo.field dialect with stencil, advect, reduce)

**Operators**:
- `field.create` — Allocate field with shape, spacing, initial value
- `field.stencil` — Apply stencil (Laplacian, gradient, divergence, custom)
- `field.advect` — Advect by velocity field (semi-Lagrangian, MacCormack, BFECC)
- `field.diffuse` — Diffusion step (Jacobi, Gauss-Seidel, CG)
- `field.project` — Pressure projection (Jacobi, multigrid, PCG)
- `field.reduce` — Reduce to scalar (sum, max, min, mean)
- `field.combine` — Element-wise combination
- `field.mask` — Apply spatial mask
- `boundary.apply` — Apply boundary conditions (periodic, clamp, reflect, noSlip)

**Dependencies**: Sparse linear algebra (for solvers), stencil patterns

**References**: `SPEC-MLIR-DIALECTS.md` (kairo.field)

---

### 1.4 Integrators Dialect

**Purpose**: Numerical integration of ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

**Why Essential**: Physics simulation, agent dynamics, and control systems all require stable, accurate time-stepping.

**Status**: 🔲 Planned (currently ad-hoc in agent operations)

**Operators**:
- `integrator.euler` — Forward Euler (1st order)
- `integrator.rk2` — Runge-Kutta 2nd order (midpoint)
- `integrator.rk4` — Runge-Kutta 4th order (classic)
- `integrator.verlet` — Velocity Verlet (symplectic, for physics)
- `integrator.leapfrog` — Leapfrog integration
- `integrator.symplectic` — Symplectic split-operator methods
- `integrator.adaptive` — Adaptive step-size (Dormand-Prince, Fehlberg)

**Dependencies**: Stochastic (for SDEs)

**Determinism**: Strict (fixed timestep), Reproducible (adaptive timestep)

---

### 1.5 Audio DSP Dialect

**Purpose**: Real-time audio synthesis, filtering, effects, mixing.

**Why Essential**: Kairo began as a creative audio kernel and must excel at low-latency, sample-accurate audio processing.

**Status**: ✅ Partially implemented (oscillators, filters, envelopes via kairo.stream)

**Operators**:
- `osc.sine` / `osc.triangle` / `osc.sawtooth` / `osc.square` — Oscillators
- `filter.lowpass` / `filter.highpass` / `filter.bandpass` / `filter.notch` — Filters
- `envelope.adsr` — Attack-Decay-Sustain-Release envelope
- `mix` — Sum multiple streams
- `amplify` — Multiply by gain
- `delay` — Delay line (circular buffer)
- `reverb` — Reverb effects (Freeverb, Schroeder, convolution)
- `compress` — Dynamic range compression
- `distortion` — Waveshaping, clipping

**Dependencies**: Transform (for spectral effects)

**References**: `SPEC-MLIR-DIALECTS.md` (kairo.stream)

---

### 1.6 Particles / Agents Dialect

**Purpose**: Particle-to-field transfers, field-to-particle forces, N-body dynamics, agent-based simulation.

**Why Essential**: Particle systems, swarm behavior, crowd simulation, and molecular dynamics all require agent operations.

**Status**: ⚙️ In progress (agent stdlib implemented, MLIR lowering planned)

**Operators**:
- `agent.spawn` — Create new agents
- `agent.remove` — Remove agents by predicate
- `agent.force_sum` — Calculate forces (brute force, grid, Barnes-Hut)
- `agent.integrate` — Update positions/velocities
- `agent.mutate` — Apply stochastic mutations
- `agent.to_field` — Deposit agent properties to field (particle-in-cell)
- `agent.from_field` — Sample field values at agent positions
- `agent.sort` — Sort by spatial locality (Morton order)

**Dependencies**: Fields (for coupling), Stochastic (for mutations), Integrators

**Determinism**: Strict (with stable ID ordering and deterministic force methods)

---

### 1.7 Visual / Fractal Dialect

**Purpose**: Fractal iteration, palette mapping, geometric warping, 2D/3D field rendering.

**Why Essential**: Creative visuals, procedural art, and scientific visualization all need efficient rendering.

**Status**: ⚙️ In progress (visual stdlib with colorization, rendering primitives)

**Operators**:
- `fractal.mandelbrot` — Mandelbrot set iteration
- `fractal.julia` — Julia set iteration
- `fractal.ifs` — Iterated function system
- `palette.apply` — Map scalar field to color palette
- `warp.displace` — Geometric displacement by vector field
- `render.points` — Render agent positions as point sprites
- `render.layers` — Composite multiple layers with blend modes
- `filter.blur` / `filter.sharpen` — Post-processing filters

**Dependencies**: Fields (for scalar/vector data), Image/Vision (for filtering)

---

## 2. Next-Wave Domains (HIGHLY LIKELY)

These domains naturally emerge once you have a computational kernel that is deterministic, multirate, type+unit safe, GPU/CPU pluggable, and graph-IR based. This is where Kairo becomes **superdomain-capable**, not just an audio/visual kernel.

---

### 2.1 Geometry & Mesh Processing

**Purpose**: Declarative geometric modeling, mesh processing, and spatial composition.

**Why Needed**: Essential for 3D modeling, CAD, robotics, physics simulation, 3D printing, computational geometry, and any domain requiring spatial reasoning.

**Status**: 🚧 In Progress (v0.9+) — **Inspired by TiaCAD v3.x**

**Key Innovation from TiaCAD**: Reference-based composition via **anchors** replaces hierarchical assemblies, making geometric composition declarative, robust, and refactor-safe.

---

#### Core Concepts (from TiaCAD)

**1. Coordinate Frames & Anchors**

Every geometric object lives in a coordinate frame and provides auto-generated anchors:

- **Frame** — Local coordinate system (origin, basis, scale)
- **Anchor** — Named reference point (`.center`, `.face_top`, `.edge_left`, etc.)
- **Placement** — Declarative composition: map anchor to anchor (not hierarchical nesting)

**Example:**
```kairo
let base = geom.box(50mm, 30mm, 5mm)
let pillar = geom.cylinder(radius=5mm, height=50mm)

# Place pillar on top of base (declarative!)
let tower = mesh.place(
    pillar,
    anchor = pillar.anchor("bottom"),
    at = base.anchor("face_top")
)
```

**Contrast with hierarchical composition:**
- ❌ Traditional: `parent.add_child(child)` → hidden state, mutation, brittle
- ✅ TiaCAD model: `place(object, anchor, at=target)` → declarative, pure, robust

**See**: `docs/SPEC-COORDINATE-FRAMES.md` for full specification

---

#### Operator Families

**2. Primitives (3D Solids)**

```kairo
geom.box(width, height, depth)
geom.sphere(radius)
geom.cylinder(radius, height)
geom.cone(radius_bottom, radius_top, height)
geom.torus(major_radius, minor_radius)
```

- All primitives auto-generate anchors (`.center`, `.face_{...}`, `.edge_{...}`)
- Deterministic (strict profile)

---

**3. Sketch Operations (2D → 2D)**

2D planar constructions (on XY plane):

```kairo
sketch.rectangle(width, height)
sketch.circle(radius)
sketch.polygon(points)
sketch.regular_polygon(n_sides, radius)

# Boolean ops on sketches
sketch.union(s1, s2, ...)
sketch.difference(s1, s2)
sketch.offset(sketch, distance)
```

---

**4. Extrusion & Revolution (2D → 3D)**

```kairo
extrude(sketch, height)
revolve(sketch, axis="z", angle=360deg)
loft(sketches, ruled=false)
sweep(profile, path, twist=0deg)
```

**Example:**
```kairo
# Create vase by revolution
let profile = sketch.polygon([(0,0), (10,0), (8,20), (5,25)])
let vase = revolve(profile, axis="y", angle=360deg)
```

---

**5. Boolean Operations (3D)**

```kairo
geom.union(s1, s2, ...)
geom.difference(s1, s2)
geom.intersection(s1, s2)

# Operator overloading
let result = solid_A + solid_B  # Union
let cut = solid_A - solid_B     # Difference
```

**Determinism**: Strict (within floating precision)

---

**6. Pattern Operations**

```kairo
pattern.linear(object, direction, count, spacing)
pattern.circular(object, axis, count, angle=360deg)
pattern.grid(object, rows, cols, spacing_x, spacing_y)
```

**Example (bolt hole pattern):**
```kairo
let hole = geom.cylinder(radius=3mm, height=10mm)
let bolts = pattern.circular(hole, axis="z", count=6)
```

---

**7. Finishing Operations**

```kairo
geom.fillet(solid, edges, radius)    # Round edges
geom.chamfer(solid, edges, distance)  # Bevel edges
geom.shell(solid, faces, thickness)   # Hollow out
```

**Example:**
```kairo
let box = geom.box(20mm, 20mm, 10mm)
let rounded = geom.fillet(box, edges=.edges(">Z"), radius=2mm)
```

---

**8. Mesh Operations (Discrete Geometry)**

```kairo
mesh.from_solid(solid, tolerance=0.01mm)
mesh.subdivide(mesh, method="catmull-clark", iterations=1)
mesh.laplacian(mesh) -> SparseMatrix
mesh.sample(mesh, field: Field<T>) -> Mesh<T>
mesh.normals(mesh) -> Mesh<Vec3>
mesh.to_field(mesh, resolution) -> Field
field.to_mesh(field, isovalue) -> Mesh  # Marching cubes
```

---

**9. Measurement & Query**

```kairo
geom.measure.volume(solid) -> f64
geom.measure.area(face) -> f64
geom.measure.bounds(object) -> BoundingBox
geom.measure.center_of_mass(solid) -> Vec3
geom.measure.distance(obj_a, obj_b) -> f64
```

---

**10. Transformations (with Explicit Origins)**

**TiaCAD principle**: All rotations/scales must specify an explicit origin (no implicit frame).

```kairo
# ✅ Explicit origin (required!)
let rotated = transform.rotate(
    mesh,
    angle = 45 deg,
    origin = mesh.anchor("center")
)

# ❌ Implicit origin (compiler error!)
let bad = transform.rotate(mesh, 45 deg)  # ERROR: origin required
```

**Transform operators:**
```kairo
transform.translate(object, offset)
transform.rotate(object, angle, axis, origin)
transform.scale(object, factor, origin)
transform.mirror(object, plane)
transform.affine(object, matrix)

# Coordinate conversions
transform.to_coord(field, coord_type="polar|spherical|cylindrical")
```

**See**: `docs/SPEC-TRANSFORM.md` Section 7 (Spatial Transformations)

---

#### Dependencies

- **Transform Dialect** — Spatial transformations, coordinate conversions
- **Fields** — For discretizations, SDF representations
- **Graph** — For mesh topology, adjacency
- **Sparse Linear Algebra** — For mesh Laplacian, PDE solvers
- **Type System** — Units (mm, m, deg, rad), frame types

---

#### Cross-Domain Integration

**Geometry → Fields (CFD, Heat Transfer)**
```kairo
let solid = geom.sphere(10mm)
let sdf = field.from_solid(solid, bounds=..., resolution=(100,100,100))
let temperature = field.solve_heat(domain=sdf, ...)
```

**Geometry → Physics (Collision, Dynamics)**
```kairo
let body = physics.rigid_body(
    shape = geom.box(10mm, 10mm, 10mm),
    mass = 1.0 kg
)
```

**Geometry → Visuals (Rendering)**
```kairo
let rendered = visual.render(
    solid,
    camera_frame = camera.frame(),
    material = material.metal(roughness=0.2)
)
```

---

#### Backend Abstraction

Geometry operations are backend-neutral. Lowering varies by backend:

| Backend | Status | Capabilities |
|---------|--------|--------------|
| **CadQuery** | Planned | Full 3D CAD (OpenCASCADE-based) |
| **CGAL** | Future | Robust booleans, mesh processing |
| **OpenCASCADE** | Future | Industrial CAD kernel |
| **GPU SDF** | Research | Implicit surfaces (GPU-friendly) |

**Backend capabilities (operator registry):**
```yaml
operator:
  name: geom.boolean.union
  backend_caps:
    cadquery: supported
    cgal: supported
    gpu_sdf: supported (implicit conversion)
```

---

#### Use Cases

- **3D Printing** — Parametric part design, STL export
- **CAD** — Mechanical design, assemblies
- **Robotics** — Robot kinematic chains, collision geometry
- **CFD** — Mesh generation for fluid simulation
- **Physics** — Collision shapes, rigid body dynamics
- **Level-Set Methods** — Implicit surface evolution
- **Computational Geometry** — Voronoi, convex hulls, mesh analysis

---

#### Testing Strategy

**1. Determinism Tests**
```kairo
# Primitives are bit-exact
assert_eq!(geom.box(10mm, 10mm, 10mm), geom.box(10mm, 10mm, 10mm))

# Anchors are deterministic
assert_eq!(box.anchor("face_top"), box.anchor("face_top"))
```

**2. Measurement Tests**
```kairo
let cube = geom.box(10mm, 10mm, 10mm)
assert_approx_eq!(geom.measure.volume(cube), 1000.0 mm³, tol=1e-9)
```

**3. Transform Tests (Explicit Origins)**
```kairo
# Rotation around center preserves center position
let rotated = transform.rotate(box, 45deg, origin=.center)
assert_vec_eq!(rotated.anchor("center").position(), box.anchor("center").position())
```

**4. Backend Equivalence**
```kairo
@backend(cadquery)
let result_cq = geom.box(...) + geom.sphere(...)

@backend(cgal)
let result_cgal = geom.box(...) + geom.sphere(...)

assert_solid_equivalent!(result_cq, result_cgal, tol=1e-6)
```

---

#### Documentation

- **`docs/SPEC-GEOMETRY.md`** — Full geometry domain specification
- **`docs/SPEC-COORDINATE-FRAMES.md`** — Frame/anchor system
- **`docs/SPEC-TRANSFORM.md`** — Section 7 (Spatial Transformations)
- **`docs/SPEC-OPERATOR-REGISTRY.md`** — Layer 6b (Geometry operators)

---

#### Summary: Why TiaCAD Matters to Kairo

TiaCAD's lessons apply **beyond geometry**:

1. **Anchors** — Unify references across domains (geometry, audio, physics, agents)
2. **Reference-based composition** — Replace hierarchies with declarative placement
3. **Explicit origins** — Prevent transform bugs, improve clarity
4. **Deterministic transforms** — Pure functions, no hidden state
5. **Backend abstraction** — Semantic operators, multiple lowering targets
6. **Parametric modeling** — Parts are pure functions (parameters → geometry)

**Key insight**: Anchors work for:
- **Geometry** — `.face_top`, `.edge_left`
- **Audio** — `.onset`, `.beat`, `.peak`
- **Physics** — `.center_of_mass`, `.joint`
- **Agents** — `.sensor`, `.waypoint`
- **Fields** — `.boundary_north`, `.gradient_max`

This unification makes Kairo's multi-domain vision coherent and practical.

---

### 2.2 Sparse Linear Algebra

**Purpose**: Operations on sparse matrices and linear systems.

**Why Needed**: Critical for PDE solvers, graph algorithms, optimization, ML kernels, simulation.

**Status**: 🔲 Planned (currently using dense linalg for small problems)

**Operators**:
- `sparse.matmul` — Sparse matrix-vector multiply
- `sparse.solve` — Solve Ax = b (iterative solvers)
- `cg` — Conjugate Gradient
- `bicgstab` — BiConjugate Gradient Stabilized
- `sparse.cholesky` — Sparse Cholesky factorization
- `csr` / `csc` — Compressed Sparse Row/Column formats
- `sparse.transpose` — Sparse matrix transpose

**Dependencies**: None (foundational)

**Use Cases**: Poisson equation, graph Laplacian, structural analysis

**MLIR Integration**: Lower to `sparse_tensor` dialect

---

### 2.3 Optimization (Convex & Non-Convex)

**Purpose**: Numerical optimization for parameter fitting, control, and learning.

**Why Needed**: Many domains rely on optimization (physics calibration, trajectory planning, ML training).

**Status**: 🔲 Planned (v0.10+)

**Operators**:
- `grad(f)` — Gradient of function
- `descent(f, lr)` — Gradient descent step
- `newton_step` — Newton's method step
- `lbfgs` — Limited-memory BFGS
- `adam` / `rmsprop` — Adaptive optimizers
- `project_to_constraint` — Project to feasible set
- `line_search` — Backtracking line search

**Dependencies**: Autodiff (for gradients), Linear Algebra

**Use Cases**: Inverse problems, control optimization, neural network training

---

### 2.4 Autodiff (Automatic Differentiation)

**Purpose**: Compute gradients, Jacobians, and Hessians automatically.

**Why Needed**: Unlocks physics simulation gradients, neural network training, differentiable graphics, differentiable audio, control optimization.

**Status**: 🔲 Planned (v0.11+)

**Operators**:
- `grad(op)` — Compute gradient of scalar function
- `jacobian` — Compute Jacobian matrix
- `hessian` — Compute Hessian matrix
- `jvp` — Jacobian-vector product (forward mode)
- `vjp` — Vector-Jacobian product (reverse mode)

**Dependencies**: None (but transforms entire graph)

**MLIR Integration**: Leverage Enzyme autodiff for MLIR

**Use Cases**: Differentiable physics, neural operators, sensitivity analysis

---

### 2.5 Graph / Network Domain

**Purpose**: Operations on graphs and networks.

**Why Needed**: Graph Laplacian transforms, spectral clustering, graph-based PDEs, network diffusion, routing/simulation, social/agent systems.

**Status**: 🔲 Planned (v0.10+)

**Operators**:
- `graph.laplacian` — Graph Laplacian matrix
- `graph.diffuse` — Diffusion on graph
- `graph.propagate` — Message propagation
- `graph.bfs` / `graph.dfs` — Breadth/depth-first search
- `graph.spectral_embed` — Spectral embedding
- `graph.pagerank` — PageRank algorithm
- `graph.shortest_path` — Dijkstra, Bellman-Ford

**Dependencies**: Sparse Linear Algebra

**Use Cases**: Social networks, circuit simulation, mesh processing

---

### 2.6 Image / Vision Ops

**Purpose**: Image processing operations (distinct from fractals and rendering).

**Why Needed**: Generic field operators + kernels for computer vision, photography, and scientific imaging.

**Status**: 🔲 Planned (v0.9+)

**Operators**:
- `blur` / `sharpen` — Convolution filters
- `edge_detect` — Sobel, Canny edge detection
- `optical_flow` — Lucas-Kanade, Farneback
- `color_transform` — RGB↔HSV, gamma correction
- `morphology.erode` / `morphology.dilate` — Morphological ops
- `histogram.equalize` — Histogram equalization
- `resize` — Image resampling (bilinear, bicubic, Lanczos)

**Dependencies**: Fields (images are 2D/3D fields), Transform (for frequency-domain filtering)

**Use Cases**: Photo processing, medical imaging, object detection

---

### 2.7 Symbolic / Algebraic Domain

**Purpose**: Symbolic manipulation, algebraic simplification, analytic transforms.

**Why Needed**: Code generation, analytic transforms, parameter solving, optimization, constraints.

**Status**: 🔲 Planned (v0.12+)

**Operators**:
- `simplify(expr)` — Algebraic simplification
- `polynomial.fit` — Polynomial fitting
- `solve.linear` — Solve linear system symbolically
- `solve.symbolic` — Symbolic equation solving
- `diff(expr, var)` — Symbolic differentiation
- `integrate(expr, var)` — Symbolic integration

**Dependencies**: May lean on SymPy or custom MLIR dialect

**Use Cases**: Automatic kernel generation, analytic Jacobians, constraint solving

---

### 2.8 I/O & Storage Providers

**Purpose**: Load/save operations for external data (images, audio, graph snapshots).

**Why Needed**: Real-world workflows require loading IR, PNGs, WAVs, saving graph snapshots, streaming big data, mmap'ed intermediates.

**Status**: 🔲 Planned (v0.9+)

**Operators**:
- `io.load` — Load file (PNG, WAV, JSON, HDF5)
- `io.save` — Save file
- `io.stream` — Stream data (real-time or batch)
- `io.query` — Query external database
- `io.mmap` — Memory-map large file

**Dependencies**: None (runtime boundary)

**Determinism**: Nondeterministic (external I/O)

**Use Cases**: Asset loading, checkpointing, live audio input

---

## 3. Advanced Domains (FUTURE EXPANSION)

These are "Version 2+" ideas — realistic but not urgent. They represent specialized use cases that extend Kairo into new application areas.

---

### 3.1 Neural Operators

**Purpose**: Neural fields, neural spectral transforms, learned PDE solvers.

**Why Interesting**: Not a "deep learning framework" — but neural fields (e.g., NeRF, SDF) and neural operators (e.g., Fourier Neural Operators) fit naturally into Kairo's field/transform model.

**Status**: 🔲 Research (v1.0+)

**Operators**:
- `mlp_field` — Neural SDF / occupancy field
- `neural_spectral` — Learned spectral transform
- `fno` — Fourier Neural Operator
- `neural_codec` — Learned audio/image compression

**Dependencies**: Autodiff, Optimization, Transform

**Use Cases**: Physics-informed ML, learned simulation, neural rendering

---

### 3.2 Probabilistic Programming

**Purpose**: Bayesian inference, sequential Monte Carlo, probabilistic models.

**Why Interesting**: Natural extension of stochastic + autodiff for probabilistic reasoning.

**Status**: 🔲 Research (v1.0+)

**Operators**:
- `sample(model)` — Sample from probabilistic model
- `condition(var, obs)` — Condition on observation
- `metropolis_step` — Metropolis-Hastings MCMC step
- `hmc_step` — Hamiltonian Monte Carlo step
- `smc.resample` — Sequential Monte Carlo resampling

**Dependencies**: Stochastic, Autodiff

**Use Cases**: Bayesian parameter estimation, uncertainty quantification, generative models

---

### 3.3 Control & Robotics

**Purpose**: Control theory operators, trajectory optimization, kinematics/dynamics.

**Why Interesting**: Kairo's deterministic semantics make it ideal for robotic control.

**Status**: 🔲 Research (v1.1+)

**Operators**:
- `pid` — PID controller
- `mpc` — Model Predictive Control
- `trajectory.optimize` — Trajectory optimization
- `kinematics.solve` — Inverse kinematics
- `robot.dynamics` — Rigid body dynamics

**Dependencies**: Fields, Integrators, Geometry, Optimization

**Use Cases**: Drone control, robotic manipulation, motion planning

---

### 3.4 Discrete Event Simulation

**Purpose**: Agent-based discrete event systems (queues, networks, processes).

**Why Interesting**: Kairo's event model already supports sample-accurate scheduling; extending to discrete event simulation is straightforward.

**Status**: 🔲 Research (v1.1+)

**Operators**:
- `queue.process` — Process queue events
- `event.route` — Route events through network
- `network.simulate` — Simulate packet routing

**Dependencies**: Stochastic (for arrival processes), Graph (for network topology)

**Use Cases**: Network simulation, supply chain modeling, epidemiology

---

## 4. Domains We Probably Won't Build

For completeness, here are domains that don't align with Kairo's mission as a **semantic transform kernel**:

- **Database / Tabular** — SQL-like queries, relational algebra (better served by databases)
- **Natural Language** — Text processing, parsing, LLMs (orthogonal to Kairo's focus)
- **Cryptography** — Hashing, encryption, signatures (security-critical, specialized)
- **Blockchain Consensus** — Proof-of-work, Byzantine agreement (niche application)
- **GUI Rendering** — Widget layout, event handling (UI frameworks handle this)

These are better addressed by specialized tools. Kairo focuses on **numerical computation, simulation, and creative coding**.

---

## Summary: Full Domain Spectrum

Here is the likely full spectrum of domains Kairo will eventually want:

### 1. Core (Must-Have) — v0.7-v0.8
| Domain | Status | Priority |
|--------|--------|----------|
| Transform | ✅ Partial | P0 |
| Stochastic | ⚙️ In Progress | P0 |
| Fields / PDE | ✅ Partial | P0 |
| Integrators | 🔲 Planned | P0 |
| Particles | ⚙️ In Progress | P0 |
| Audio DSP | ✅ Partial | P0 |
| Visual / Fractal | ⚙️ In Progress | P0 |

### 2. Next Wave (Highly Likely) — v0.9-v1.0
| Domain | Status | Priority |
|--------|--------|----------|
| Geometry/Mesh | 🔲 Planned | P1 |
| Sparse Linear Algebra | 🔲 Planned | P1 |
| Optimization | 🔲 Planned | P1 |
| Autodiff | 🔲 Planned | P1 |
| Graph/Network | 🔲 Planned | P1 |
| Image/Vision | 🔲 Planned | P1 |
| Symbolic/Algebraic | 🔲 Planned | P2 |
| I/O & Storage | 🔲 Planned | P1 |

### 3. Advanced Future — v1.1+
| Domain | Status | Priority |
|--------|--------|----------|
| Neural Operators | 🔲 Research | P3 |
| Probabilistic Programming | 🔲 Research | P3 |
| Control & Robotics | 🔲 Research | P3 |
| Discrete Event Simulation | 🔲 Research | P3 |

**Legend**:
- ✅ Partial: Implemented but incomplete
- ⚙️ In Progress: Active development
- 🔲 Planned: Design phase
- 🔲 Research: Exploratory

---

## Design Principles

All Kairo domains adhere to these principles:

1. **Deterministic by Default** — Operations are reproducible unless explicitly marked `@nondeterministic`
2. **Type + Unit Safe** — Physical units are tracked and validated at compile time
3. **Multirate Scheduling** — Different domains can run at different rates (audio, control, visual)
4. **GPU/CPU Pluggable** — Operations lower to MLIR and can run on any backend
5. **Minimal, Sharply Defined** — Each domain has a focused scope; lower to standard dialects ASAP
6. **Extensible** — New operators can be added without breaking existing code

---

## Integration Example: Multi-Domain Simulation

A realistic Kairo program using multiple domains:

```kairo
scene FluidWithParticles {
  // Fields: Velocity and pressure
  let velocity: Field2D<Vec2<m/s>> = field.create(512, 512, Vec2(0, 0))
  let pressure: Field2D<Pa> = field.create(512, 512, 0Pa)

  // Agents: Particles advected by fluid
  let particles: Agents<{pos: Vec2<m>, color: Vec3}> = agent.create(1000)

  step(dt: Time) {
    // Stochastic: Add random force
    let force_field = stochastic.perlin_noise(velocity.shape, seed=42)

    // Fields: Advect, diffuse, project velocity
    velocity = field.advect(velocity, velocity, dt, method="BFECC")
    velocity = field.diffuse(velocity, viscosity=0.01, dt, solver="CG")
    velocity = field.project(velocity, dt, solver="multigrid")

    // Particles: Update positions from velocity field
    particles = agent.from_field(particles, velocity, "velocity")
    particles = agent.integrate(particles, dt, method="RK4")

    // Image: Render particles to field
    let density = agent.to_field(particles, field.shape, "density")

    // Visual: Colorize and render
    let color_field = palette.apply(density, palette="viridis")
    out visual = render.field(color_field)

    // Audio: Sonify pressure field
    let pressure_sample = field.reduce(pressure, "mean")
    let tone = osc.sine(pressure_sample * 100Hz)
    out audio = tone
  }
}
```

**Domains Used**:
1. **Fields** — Fluid velocity and pressure
2. **Stochastic** — Perlin noise forcing
3. **Particles** — Advected by fluid
4. **Integrators** — RK4 time-stepping
5. **Image** — Particle-to-field rasterization
6. **Visual** — Palette mapping and rendering
7. **Audio** — Sonification via oscillator

This demonstrates Kairo's **cross-domain composability** — all domains share the same type system, scheduler, and MLIR backend.

---

## Roadmap Implications

### v0.8 (Current → Next Release)
- **Complete Core Domains**: Finish Stochastic, Integrators, Particles
- **MLIR Lowering**: All core dialects lower to LLVM/GPU
- **Conformance Tests**: Determinism guarantees for all core ops

### v0.9-v0.10 (Next Wave Phase 1)
- **Add**: Geometry/Mesh, Sparse Linear Algebra, I/O & Storage
- **Focus**: 3D simulation, large-scale PDEs, asset loading

### v1.0 (Next Wave Phase 2)
- **Add**: Optimization, Autodiff, Graph/Network, Image/Vision
- **Focus**: Differentiable programming, ML integration, vision pipelines

### v1.1+ (Advanced Domains)
- **Explore**: Neural Operators, Probabilistic Programming, Control/Robotics
- **Focus**: Research applications, novel use cases

---

## Cross-Cutting Concerns

### Determinism Across Domains
All domains support three determinism tiers:
1. **Strict** — Bit-identical (e.g., `field.diffuse`, `agent.force_sum` with deterministic methods)
2. **Reproducible** — Deterministic within precision (e.g., iterative solvers)
3. **Nondeterministic** — External I/O or adaptive termination (e.g., `io.stream(live)`)

### MLIR Dialect Strategy
- **Domain-Specific Dialects**: kairo.stream, kairo.field, kairo.transform, kairo.schedule
- **Lower to Standard Dialects ASAP**: linalg, affine, vector, arith, math, scf, memref
- **Backend Dialects**: llvm (CPU), gpu (CUDA/ROCm), spirv (Vulkan)

See `SPEC-MLIR-DIALECTS.md` for current dialect definitions.

### GPU Acceleration
All domains follow Kairo's GPU lowering principles:
- Structured parallelism (explicit iteration spaces)
- Memory hierarchy management (global/shared/register)
- Static shape preference
- Warp-friendly execution
- Deterministic GPU semantics

See `GPU_MLIR_PRINCIPLES.md` for details.

---

## Conclusion

Kairo's domain architecture is designed for **long-term extensibility** while maintaining **core simplicity**. By focusing on:
- Deterministic semantics
- Type + unit safety
- Multirate scheduling
- MLIR-based compilation
- GPU/CPU portability

...we create a foundation that naturally supports audio, graphics, physics, AI, and beyond — all in a single unified system.

This document will evolve as new domains are designed, prototyped, and integrated. It serves as both a **vision** and a **contract**: every domain must justify its existence and integrate coherently with the rest of the system.

---

## References

### Core Specifications
- **SPEC-MLIR-DIALECTS.md** — Current dialect definitions (kairo.stream, kairo.field, kairo.transform, kairo.schedule)
- **architecture.md** — Overall system architecture
- **GPU_MLIR_PRINCIPLES.md** — GPU lowering design rules
- **SPEC-TYPE-SYSTEM.md** — Type system and unit tracking
- **SPEC-SCHEDULER.md** — Multirate scheduling semantics
- **SPEC-OPERATOR-REGISTRY.md** — Operator metadata and registration
- **SPEC-COORDINATE-FRAMES.md** — Unified frame and anchor system
- **SPEC-GEOMETRY.md** — Geometry domain specification (TiaCAD patterns)

### Architectural Decision Records
- **ADR/001-unified-reference-model.md** — Decision on unified reference system
- **ADR/002-cross-domain-architectural-patterns.md** — Patterns from TiaCAD, RiffStack, and Strudel

### Implementation Guides
- **GUIDES/DOMAIN_IMPLEMENTATION_GUIDE.md** — Step-by-step domain implementation guide
- **LEARNINGS/OPERATOR_REGISTRY_EXPANSION.md** — Detailed operator catalogs for 7 priority domains

---

**End of Document**
