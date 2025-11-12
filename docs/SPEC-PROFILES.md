# SPEC: Profiles

Profiles are **policy bundles** that control determinism, precision, and solver behavior across Kairo programs. They centralize configuration that would otherwise clutter individual operations.

## Profile Definition

```kairo
profile strict {
  determinism = "bitexact"
  precision = f64
  project.method = "cg"
  project.iter = 40
  project.tol = 1e-8
}

profile medium {
  determinism = "repro"
  precision = f32
  project.method = "cg"
  project.iter = 20
  project.tol = 1e-4
}

profile live {
  determinism = "replayable"
  precision = f16
  project.method = "jacobi"
  project.iter = 10
}
```

## Profile Attributes

### Determinism Levels

- `bitexact` — Bitwise-identical results across platforms, runs, and compiler versions.
- `repro` — Reproducible within a platform/compiler (may vary across hardware).
- `replayable` — Deterministic replay from recorded inputs/events (numerical drift allowed).
- `best_effort` — Non-deterministic (e.g., for performance or real-time constraints).

### Precision

- `f64` — Double precision
- `f32` — Single precision
- `f16` — Half precision
- `mixed` — Mixed precision (solver-dependent)

### Solver Aliases

Profiles can define default parameters for solver families:

```kairo
profile gpu_optimized {
  determinism = "repro"
  precision = f32

  # Solver defaults
  project.method = "pcg"
  project.preconditioner = "jacobi"
  project.iter = 30

  diffuse.method = "jacobi"
  diffuse.iter = 5

  fft.library = "cufft"
}
```

## Precedence Rules

Profiles follow a **strict precedence hierarchy**:

1. **Per-operation override** (highest precedence)
2. **Solver-specific alias** (from active profile)
3. **Module-level profile**
4. **Global profile** (lowest precedence)

### Example

```kairo
# Global profile (precedence 4)
profile global = strict

# Module profile (precedence 3)
profile module = medium

flow sim(dt=0.01) @profile=module {
  # Uses module profile (medium): f32, 20 iterations
  vel = project(vel)

  # Solver alias from profile (precedence 2)
  density = diffuse(density, rate=0.1, dt)

  # Per-op override (precedence 1, highest)
  pressure = project(pressure, method="mg", iter=50, tol=1e-6)
}
```

## Profile Application

### Global Profile

Set at program/session level:

```bash
kairo run sim.kairo --profile=strict
```

Or in code:

```kairo
@global_profile = strict
```

### Module Profile

Applies to a flow or module:

```kairo
flow physics(dt=0.01) @profile=strict {
  # All operations in this flow use 'strict' profile
  vel = navier_stokes(vel, dt)
}

flow rendering(dt=1/60) @profile=live {
  # Relaxed profile for real-time rendering
  output colorize(density)
}
```

### Per-Operation Override

```kairo
# Override profile for this specific call
vel = project(vel, precision=f64, determinism="bitexact", iter=100)
```

## Determinism Enforcement

The compiler **enforces** determinism levels:

- `bitexact`: Requires:
  - Deterministic RNG with explicit seeds
  - No platform-specific intrinsics (unless proven deterministic)
  - No async I/O within flow (use barriers)
  - Strict floating-point modes (`-fp-model strict`)

- `repro`: Requires:
  - Deterministic RNG
  - Reproducible reduction orders (e.g., Kahan summation)

- `replayable`: Requires:
  - Event stream recording

The compiler **rejects** code that violates its profile's determinism level.

### Example: Determinism Error

```kairo
profile strict_mode {
  determinism = "bitexact"
}

flow sim(dt=0.01) @profile=strict_mode {
  # ERROR: random() without seed violates bitexact determinism
  let noise = random()  // ❌ Compiler error

  # OK: explicit seed
  let noise = random(seed=42)  // ✅
}
```

## Backend Selection

Profiles influence **backend selection** during lowering:

```kairo
profile cpu_vectorized {
  precision = f32
  backend.target = "cpu"
  backend.vector_width = 8  # AVX2
}

profile gpu_compute {
  precision = f16
  backend.target = "gpu"
  backend.api = "vulkan"  # or "metal", "cuda"
}
```

The MLIR lowering pipeline selects appropriate dialects and passes based on the active profile.

## Introspection

Profiles are **introspectable** at runtime:

```kairo
let prof = introspect_profile()
print(prof.determinism)  // "bitexact"
print(prof.precision)    // f64
```

## Standard Library Profiles

Kairo ships with standard profiles:

- `strict` — Maximum determinism and precision
- `balanced` — Good determinism with f32
- `live` — Real-time/interactive use
- `gpu` — GPU-optimized (may sacrifice determinism)
- `debug` — Debug symbols, verbose errors, slower

Users can define custom profiles and share them via modules.

## Migration from v0.3.1

Current per-op `deterministic=true` flags are **deprecated** in favor of profiles:

```kairo
# v0.3.1 (old)
vel = project(vel, deterministic=true, precision="f64", max_iterations=40)

# v1.0 (new)
profile strict { determinism = "bitexact", precision = f64, project.iter = 40 }
vel = project(vel) @profile=strict
```

Old syntax remains supported via compatibility aliases.

## Examples

### Research Simulation (Bitexact)

```kairo
profile research {
  determinism = "bitexact"
  precision = f64
  project.iter = 100
  project.tol = 1e-12
}

flow pde(dt=0.001) @profile=research {
  temp = diffuse(temp, rate=KAPPA, dt)
  vel = project(vel)
  density = advect(density, vel, dt)
}
```

### Live Audio (Real-Time)

```kairo
profile audio_live {
  determinism = "replayable"
  precision = f32
  fft.library = "pffft"  # Fast, platform-optimized
}

flow synth(dt=1/48000) @profile=audio_live {
  let osc = sine(freq=440Hz)
  let filtered = lpf(osc, cutoff=2000Hz)
  output filtered
}
```

### GPU Rendering (Best Effort)

```kairo
profile gpu_render {
  determinism = "best_effort"
  precision = f16
  backend.target = "gpu"
  backend.api = "metal"
}

flow render(dt=1/60) @profile=gpu_render {
  let spectrum = fft(audio_buffer)
  output colorize(spectrum, palette="rainbow")
}
```

## Open Questions

1. **Profile Composition**: Can profiles inherit/extend each other?
2. **Runtime Switching**: Can profiles change mid-execution (e.g., "strict mode" for critical sections)?
3. **Library Contracts**: How do third-party libraries declare their determinism guarantees?
