# Kairo Kernel Architecture (v1.0 Draft)

> "LLVM gives you computation. MLIR gives you structure. **Kairo gives you meaning.**"

Kairo is a **semantic kernel** for deterministic creative and scientific computation. It sits between domain DSLs (audio/visual/astro/physics) and execution backends (MLIR→LLVM/GPU/Async). The kernel defines the invariants of **time, space, rate, units, state, profiles, boundaries, and interfaces**, and provides a deterministic scheduler and introspection.

## 1. Role in the Stack

```
DSLs → **Kairo Kernel** → MLIR dialects → LLVM/SPIR-V/Metal/Audio
```

- **Above**: StreamTone, Luma, Asterion, etc.
- **Within**: typed Streams & Events, Spaces, Grid, Boundaries, Interfaces, Profiles
- **Below**: MLIR (`linalg/affine/vector/gpu/async`), vendor toolchains, scientific libs

## 2. Kernel Services

- **Semantic**: units/dimensions; space/time/rate; boundary/interface contracts; deterministic time model; explicit state.
- **Structural**: flow DAG IR; multirate scheduler; state versioning; profile overlays; cross-domain linking.
- **Scientific**: field algebra; solver registry; grid spacing/centering; deterministic RNG; numerical profiles.
- **Creative**: hot-reload; introspection; recording & replay; visualization hooks; annotations.

## 3. Core Types

- `Stream<T,Domain,Rate>` — unified supertype for signals, fields, images, etc.
- `Evt<A>` — timestamped event sequence (strict replay).
- `Space` + `Grid` — dimensional metadata (spacing, centering).
- `Boundary`, `Interface` — typed couplers (Dirichlet/Neumann/periodic/reflect; domain interfaces).
- `Profile` — determinism & precision policy bundle.

Aliases:
- `Signal<T>` ≡ `Stream<T,1D,audio|control>`
- `FieldND<T>` ≡ `Stream<T,ND,sim>`

## 4. Deterministic Multirate Scheduler

- Rates: `{audio, control, visual, sim}` with explicit `dt` or `sample_rate`.
- Execution: partition time by LCM of steps; enforce sample-accurate event fences.
- Conversions: `resample(stream, to=rate, mode=nearest|linear|cubic)` (deterministic rounding).
- State: double-buffered updates with step boundaries; no hidden globals.

## 5. Profiles

```text
profile strict: { determinism=bitexact, precision=f64, solver=CG(iter=40) }
profile medium: { determinism=repro, precision=f32 }
profile live:   { determinism=replayable, precision=f16 }
```

Precedence: per-op > solver alias > module profile > global profile.

## 6. Boundary & Interface Contracts

- **Boundary**: face/region rules (reflect, periodic, no-slip, inflow(expr), outflow, custom).
- **Interface**: multi-domain coupling (continuous, flux_match, insulated, custom).

Solvers declare `BoundaryContract{apply=pre|post, mode=ghost|mirror}` to fix ordering.

## 7. Lowering

**Kairo IR → MLIR:**

- Streams/Fields → `linalg/affine/vector`
- GPU kernels → `gpu`
- Async/IO → `async`, host callbacks
- Audio device sink → async push with deterministic blocks

**MLIR → LLVM, SPIR-V, Metal**

## 8. Introspection

- `introspect(graph)` → JSON graph (nodes, edges, rates, profiles).
- `visualize(graph|boundary|stream)` → debug overlays.

## 9. Extension API

Register dialects with ops and default rates; point them to kernel services and lowering passes.
