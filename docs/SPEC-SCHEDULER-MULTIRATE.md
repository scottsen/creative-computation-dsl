# SPEC: Deterministic Multirate Scheduler

## Rates

Kairo supports multiple execution rates within a single program:

- `audio` (e.g., 48000 Hz) — Audio-rate processing
- `control` (e.g., 100 Hz) — Control signals and automation
- `visual` (e.g., 60 Hz) — Frame-rate rendering
- `sim` (user `dt`, e.g., 0.01 s) — Simulation timestep

## Execution Model

The scheduler executes flows deterministically across multiple rates:

1. **LCM Partitioning**: Compute the least common multiple (LCM) of all active rates to determine partition boundaries.
2. **Topological Order**: Within each partition, execute flows in topological order based on data dependencies.
3. **Sample-Accurate Timing**: All rate transitions and event deliveries occur at exact partition boundaries.

## Example: Mixed-Rate Flow

```kairo
# Audio-rate synthesis
flow synth(dt=1/48000 @rate=audio) {
  let osc = sine(freq=440Hz)
  output osc
}

# Control-rate modulation
flow control(dt=1/100 @rate=control) {
  let lfo = sine(freq=2Hz)
  output lfo
}

# Visual-rate rendering
flow render(dt=1/60 @rate=visual) {
  let spectrum = fft(audio_stream)
  output colorize(spectrum)
}
```

### Execution Timeline

```
Time:     0.0s      0.01s     0.02s     0.03s
          |         |         |         |
audio:    ████████████████████████████████  (480 samples per control tick)
control:  ██████████|         |         |
visual:   ██████████████████|         |
```

## Resampling

Cross-rate operations require explicit resampling:

```kairo
resample(stream, to=rate, mode=nearest|linear|cubic)
```

### Modes

- `nearest`: Nearest-neighbor (zero-order hold)
- `linear`: Linear interpolation
- `cubic`: Cubic spline interpolation

All modes use **deterministic rounding** with well-defined behavior at boundaries.

### Example

```kairo
flow audio(dt=1/48000 @rate=audio) {
  # Resample control-rate LFO to audio rate
  let lfo_audio = resample(lfo_control, to=audio, mode=linear)
  let modulated = sine(freq=440Hz + lfo_audio * 50Hz)
  output modulated
}
```

## Events and Fences

Event streams (`Evt<A>`) create **sample-accurate fences** in the scheduler:

```kairo
let notes = event.from_list([
  (0.0s, Note{freq: 440Hz}),
  (0.5s, Note{freq: 554Hz})
])

flow audio(dt=1/48000 @rate=audio) {
  # Event delivered at exact sample: floor(0.5s * 48000Hz) = 24000
  let gate = notes.to_signal(mode=impulse)
  output gate
}
```

### Event Delivery Guarantees

- Events are delivered **exactly once** at the partition containing the event time.
- If multiple events occur within the same partition, they are processed in **timestamp order**.
- Cross-rate event delivery uses the same deterministic rounding as `resample()`.

## Scheduler Algorithm (Pseudocode)

```python
def schedule(flows, duration):
  rates = collect_rates(flows)
  partition_size = lcm(rates)

  t = 0.0
  while t < duration:
    partition_end = t + partition_size

    # Collect events in this partition
    events = collect_events(t, partition_end)

    # Execute flows in topological order
    for flow in topological_sort(flows):
      if should_tick(flow, t, partition_end):
        execute_flow(flow, events)

    # Update state buffers
    commit_state_updates()

    t = partition_end
```

## State Updates

State variables (`@state`) are **double-buffered**:

- Reads within a partition see the state from the **beginning** of the partition.
- Writes are accumulated and **committed** at partition boundaries.
- This ensures deterministic execution regardless of intra-partition ordering.

```kairo
@state counter: i32 = 0

flow tick1(dt=0.01 @rate=control) {
  counter = counter + 1  // Buffered write
}

flow tick2(dt=0.01 @rate=control) {
  let x = counter        // Reads value from partition start
  output x
}
```

## Boundary Contracts

When flows involve solvers with boundary conditions, the scheduler applies `BoundaryContract` rules:

```kairo
solver.declare_contract(BoundaryContract{
  apply: pre,           // Apply boundary conditions before solver
  mode: ghost           // Use ghost cells
})
```

The scheduler inserts boundary application at the appropriate point in the execution order.

## Determinism Guarantees

- **Bitexact Replay**: Given identical inputs and seeds, execution produces bitwise-identical results.
- **Platform Independence**: Results are identical across platforms (with appropriate profile settings).
- **Rate Independence**: Changing rates does not affect the logical result (modulo numerical precision).

## Open Questions

1. **Adaptive Timesteps**: How should the scheduler handle flows with adaptive `dt`?
2. **Real-Time Scheduling**: Priority/latency guarantees for live audio?
3. **Distributed Execution**: Can we partition across multiple cores/machines while preserving determinism?
