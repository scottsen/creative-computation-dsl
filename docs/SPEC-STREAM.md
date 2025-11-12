# SPEC: Streams & Events

## Stream Supertype

```text
Stream<T, Domain, Rate>

  Domain: 0D|1D|2D|3D (scalars/signals/fields/images)
  Rate: audio|control|visual|sim with explicit dt or sample rate.
```

Aliases remain: `Signal<T>`, `FieldND<T>`.

## Core Ops

```kairo
map(fn)
combine(b, fn)
integrate(rate, dt)
resample(to, mode)
to_domain(newDomain, transform)
```

## Event Streams

```text
Evt<A> = [(time: f32[s], value: A)]  // ordered, unique times
```

### Ops

```kairo
event.from_list([...])
event.map(fn)
event.merge(e1, e2, stable=true)
event.to_signal(mode=hold|impulse)  // sample-accurate fences
```

### Determinism

Strict ordering; replay uses recorded list verbatim.

## Examples

### Basic Stream Operations

```kairo
# Unified stream type
let signal: Stream<f32, 1D, audio> = sine(440Hz)
let field: Stream<f32, 2D, sim> = random_normal(seed=42)

# Alias types (backwards compatible)
let sig: Signal<f32> = sine(440Hz)
let fld: Field2D<f32> = random_normal(seed=42)

# Cross-rate operations require explicit resampling
let control_sig = resample(signal, to=control, mode=linear)
```

### Event Stream Examples

```kairo
# Create event stream from list
let notes = event.from_list([
  (0.0s, Note{freq: 440Hz, vel: 0.8}),
  (0.5s, Note{freq: 554Hz, vel: 0.6}),
  (1.0s, Note{freq: 660Hz, vel: 0.7})
])

# Map over events
let transposed = notes.map(|n| Note{freq: n.freq * 1.5, vel: n.vel})

# Merge two event streams (stable ordering)
let combined = event.merge(notes1, notes2, stable=true)

# Convert to signal with sample-accurate timing
let gate = notes.to_signal(mode=impulse)
let held = notes.to_signal(mode=hold)
```

## Type Relationships

```
Stream<T,Domain,Rate>
  ├─ Stream<T,0D,R>     ≡ Scalar<T> @ Rate R
  ├─ Stream<T,1D,audio> ≡ Signal<T>
  ├─ Stream<T,1D,control> ≡ Control<T>
  ├─ Stream<T,2D,sim>   ≡ Field2D<T>
  └─ Stream<T,3D,sim>   ≡ Field3D<T>

Evt<A>                  (distinct: timestamped events)
```

## Rate Semantics

- `audio`: Fixed sample rate (e.g., 48000 Hz)
- `control`: Control rate (e.g., 100 Hz)
- `visual`: Frame rate (e.g., 60 Hz)
- `sim`: Simulation timestep (variable dt)

All rates must be explicitly declared or inferred from context. Cross-rate operations require `resample()`.
