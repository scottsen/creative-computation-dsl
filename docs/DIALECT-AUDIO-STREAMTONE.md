# Dialect: Kairo.Audio (StreamTone)

**StreamTone** is the audio-focused dialect for Kairo, providing ergonomic DSL syntax for sound synthesis, processing, and composition while lowering to Kairo's kernel types and scheduler.

## Goal

Create an **expressive, composable audio language** that:
- Lowers to `Stream<T,Domain,Rate>` and multirate scheduler
- Provides sample-accurate event handling via `Evt<A>`
- Supports both synthesis and physical modeling
- Integrates seamlessly with Kairo's field/visual/sim dialects
- Maintains determinism guarantees

## Type Aliases

StreamTone uses ergonomic aliases for Kairo kernel types:

```kairo
Sig  ≡ Stream<f32, 1D, audio>      # Audio-rate signal
Ctl  ≡ Stream<f32, 0D, control>    # Control-rate scalar
Evt<A>                              # Event stream (unchanged)
```

## Library Structure

### 1. Oscillators (Sampling)

```kairo
sine(freq: Ctl) -> Sig
saw(freq: Ctl) -> Sig
square(freq: Ctl, duty: Ctl = 0.5) -> Sig
tri(freq: Ctl) -> Sig
noise(seed: u32) -> Sig
```

**Example:**

```kairo
flow audio(dt=1/48000 @rate=audio) {
  let osc = sine(freq=440Hz)
  output osc
}
```

### 2. Envelopes

```kairo
adsr(attack: f32[s], decay: f32[s], sustain: f32, release: f32[s], gate: Evt<Bang>) -> Sig
ar(attack: f32[s], release: f32[s], gate: Evt<Bang>) -> Sig
envexp(tau: f32[s], gate: Evt<Bang>) -> Sig
```

**Example:**

```kairo
let gate = event.from_list([at(0s, bang), at(1s, bang)])
let env = adsr(
  attack=0.01s,
  decay=0.1s,
  sustain=0.7,
  release=0.3s,
  gate=gate
)
```

### 3. Filters

```kairo
lpf(input: Sig, cutoff: Ctl, resonance: Ctl = 0.0) -> Sig
hpf(input: Sig, cutoff: Ctl, resonance: Ctl = 0.0) -> Sig
bpf(input: Sig, center: Ctl, bandwidth: Ctl) -> Sig
svf(input: Sig, cutoff: Ctl, resonance: Ctl) -> (Sig, Sig, Sig)  # low, band, high
```

**Example:**

```kairo
let osc = saw(freq=220Hz)
let filtered = lpf(osc, cutoff=1000Hz, resonance=0.7)
```

### 4. Effects

```kairo
delay(input: Sig, time: Ctl, feedback: Ctl) -> Sig
chorus(input: Sig, rate: Ctl, depth: Ctl, mix: Ctl) -> Sig
flanger(input: Sig, rate: Ctl, depth: Ctl, feedback: Ctl) -> Sig
reverb(input: Sig, room_size: Ctl, damping: Ctl, mix: Ctl) -> Sig
conv(input: Sig, ir: string) -> Sig  # Convolution with impulse response
```

**Example:**

```kairo
let dry = sine(freq=440Hz)
let wet = reverb(dry, room_size=0.8, damping=0.5, mix=0.3)
output wet
```

### 5. Utilities

```kairo
mix(a: Sig, b: Sig, balance: Ctl = 0.5) -> Sig
pan(input: Sig, position: Ctl) -> (Sig, Sig)  # stereo
db2lin(db: Ctl) -> Ctl
lin2db(lin: Ctl) -> Ctl
clip(input: Sig, min: f32, max: f32) -> Sig
```

**Example:**

```kairo
let osc1 = sine(freq=440Hz)
let osc2 = saw(freq=220Hz)
let mixed = mix(osc1, osc2, balance=0.3)
let (left, right) = pan(mixed, position=0.2)
output stereo(left, right)
```

## Physical Modeling

StreamTone includes primitives for **physical modeling synthesis**:

### Waveguide String

```kairo
string(freq: Ctl, excitation: Sig, t60: f32[s]) -> Sig
```

**Parameters:**
- `freq`: Fundamental frequency
- `excitation`: Input signal (e.g., noise burst)
- `t60`: Decay time (time to -60dB)

**Example:**

```kairo
let gate = event.from_list([at(0s, bang)])
let exc = noise(seed=42) |> lpf(cutoff=8000Hz) |> envexp(tau=0.02s, gate=gate)
let tone = string(freq=220Hz, excitation=exc, t60=2.0s)
output tone
```

### Body Resonance

```kairo
bodyIR(input: Sig, ir_file: string) -> Sig
```

**Example:**

```kairo
let plucked = string(freq=220Hz, excitation=exc, t60=1.5s)
let body = bodyIR(plucked, "guitar_body.wav")
output body
```

### Pickup and Amp Modeling

```kairo
pickup(input: Sig, position: f32, type: string) -> Sig  # "single_coil", "humbucker"
amp(input: Sig, gain: Ctl, tone: Ctl, model: string) -> Sig  # "clean", "crunch", "lead"
cab(input: Sig, model: string) -> Sig  # Cabinet simulation
```

**Complete Chain Example:**

```kairo
flow guitar(dt=1/48000 @rate=audio) {
  let gate = event.from_list([at(0s, bang)])

  # Excitation
  let exc = noise(seed=1) |> lpf(8000Hz) |> envexp(0.02s, gate)

  # String
  let str = string(freq=220Hz, excitation=exc, t60=1.5s)

  # Body resonance
  let body = bodyIR(str, "acoustic_body.ir")

  # Pickup
  let picked = pickup(body, position=0.7, type="single_coil")

  # Amp
  let amped = amp(picked, gain=0.6, tone=0.5, model="crunch")

  # Cabinet
  let final = cab(amped, model="1x12_open_back")

  output final
}
```

## Events & Score

StreamTone provides **sample-accurate event handling**:

```kairo
# Event creation
event.from_list([(time, value), ...])
event.from_midi(file: string)

# Score DSL (future)
score([
  note(0.0s, freq=440Hz, vel=0.8, dur=0.5s),
  note(0.5s, freq=554Hz, vel=0.6, dur=0.5s),
  note(1.0s, freq=660Hz, vel=0.7, dur=0.5s)
])
```

**Example:**

```kairo
struct Note {
  freq: f32[Hz]
  vel: f32
}

let notes = event.from_list([
  (0.0s, Note{freq: 440Hz, vel: 0.8}),
  (0.5s, Note{freq: 554Hz, vel: 0.6}),
  (1.0s, Note{freq: 660Hz, vel: 0.7})
])

flow audio(dt=1/48000 @rate=audio) {
  # Convert events to control signal (hold)
  let freq_ctl = notes.map(|n| n.freq).to_signal(mode=hold)
  let vel_ctl = notes.map(|n| n.vel).to_signal(mode=hold)

  # Gate on note events
  let gate = notes.to_signal(mode=impulse)

  # Synth
  let osc = sine(freq=freq_ctl)
  let env = adsr(0.01s, 0.1s, 0.7, 0.3s, gate)
  output osc * env * vel_ctl
}
```

## Lowering to Kairo Kernel

StreamTone lowers to Kairo's core primitives:

### Signal Nodes → Kairo Kernels

```kairo
# StreamTone
let osc = sine(freq=440Hz)

# Lowers to Kairo kernel:
@state phase: f32 = 0.0

flow audio(dt=1/48000 @rate=audio) {
  phase = (phase + 440.0 * dt) % 1.0
  let osc = sin(phase * 2.0 * pi)
  output osc
}
```

### Filters → Linalg Ops

```kairo
# StreamTone
let filtered = lpf(input, cutoff=1000Hz, resonance=0.5)

# Lowers to biquad (MLIR linalg ops for SIMD)
```

### Convolution → FFT Backend

```kairo
# StreamTone
let reverb = conv(input, "hall.ir")

# Lowers to:
# - FFT (via MLIR or external lib)
# - Element-wise multiply
# - IFFT
# - Overlap-add
```

### Events → Scheduler Fences

```kairo
# StreamTone event
let gate = event.from_list([at(0.5s, bang)])

# Lowers to scheduler fence at sample 24000 (for 48kHz)
# Kernel receives event value at exact sample
```

## Audio Output Sink

StreamTone outputs are routed to **audio device sinks** via async:

```kairo
flow audio(dt=1/48000 @rate=audio) {
  let sig = sine(freq=440Hz)
  output sig  # Routes to audio device
}
```

**Lowering:**
- Kairo kernel produces audio buffers (deterministic block size)
- MLIR `async` dialect pushes buffers to device queue
- Device driver handles real-time scheduling

## Integration with Other Dialects

StreamTone integrates seamlessly with Kairo's other dialects:

### Audio + Visual

```kairo
flow audio(dt=1/48000 @rate=audio) {
  @state spectrum: Field1D<f32> = zeros(512)

  let osc = sine(freq=440Hz)
  spectrum = fft(osc, size=512)

  output osc  # Audio
}

flow visual(dt=1/60 @rate=visual) {
  # Resample audio-rate spectrum to visual-rate
  let spec_vis = resample(spectrum, to=visual, mode=linear)

  output colorize(spec_vis, palette="rainbow")
}
```

### Audio-Driven Simulation

```kairo
flow audio(dt=1/48000 @rate=audio) {
  let bass = sine(freq=60Hz) |> lpf(200Hz)
  output bass
}

flow sim(dt=0.01 @rate=sim) {
  # Resample audio to control fluid simulation
  let bass_energy = resample(bass, to=sim, mode=rms)

  @state vel: Field2D<Vec2<f32>> = zeros((256, 256))
  vel = advect(vel, vel, dt)
  vel = vel + force_from_audio(bass_energy)

  output colorize(vel)
}
```

## Determinism

StreamTone inherits Kairo's determinism guarantees:

- **Bitexact oscillators** (via profile)
- **Deterministic filters** (fixed-point or strict FP modes)
- **Sample-accurate events** (exact replay)
- **Deterministic FFT** (via profiled library choice)

**Example:**

```kairo
profile strict_audio {
  determinism = "bitexact"
  precision = f64
  fft.library = "fftw3"  # Deterministic
}

flow audio(dt=1/48000 @rate=audio) @profile=strict_audio {
  let osc = sine(freq=440Hz)
  output osc
  # ✅ Bitwise-identical across platforms
}
```

## Examples

### 1. FM Synthesis

```kairo
flow fm_synth(dt=1/48000 @rate=audio) {
  let mod = sine(freq=5Hz) * 50Hz
  let carrier = sine(freq=440Hz + mod)

  let env = ar(attack=0.01s, release=0.5s, gate=gate)
  output carrier * env
}
```

### 2. Subtractive Synth

```kairo
flow subtractive(dt=1/48000 @rate=audio) {
  let osc = saw(freq=110Hz)
  let filtered = lpf(osc, cutoff=lfo * 2000Hz + 500Hz, resonance=0.8)

  let lfo = sine(freq=2Hz) * 0.5 + 0.5
  let env = adsr(0.01s, 0.2s, 0.5, 0.3s, gate)

  output filtered * env
}
```

### 3. Physical Model (Plucked String)

```kairo
use audio  # Kairo.Audio (StreamTone)

flow plucked(dt=1/48000 @rate=audio) {
  let gate = event.from_list([at(0s, bang)])
  let freq = 220Hz

  # Excitation
  let exc = noise(seed=1) |> lpf(8000Hz) |> envexp(0.02s, gate)

  # String
  let str = string(freq=freq, excitation=exc, t60=1.5s)

  # Body resonance
  let body = bodyIR(str, "acoustic_body.ir")

  # Add reverb
  let final = reverb(body, room_size=0.5, damping=0.6, mix=0.2)

  output stereo = pan(final, position=0.1)
}
```

## Roadmap

### Phase 1: Core Library (Milestone 3)
- [ ] Oscillators (sine, saw, square, tri, noise)
- [ ] Basic filters (lpf, hpf, bpf)
- [ ] ADSR envelope
- [ ] Mix, pan, clip utilities

### Phase 2: Effects & Physical Modeling
- [ ] Delay, reverb, chorus, flanger
- [ ] Convolution (FFT backend)
- [ ] Waveguide string
- [ ] Body/pickup/amp/cab models

### Phase 3: Advanced Features
- [ ] MIDI event import
- [ ] Score DSL
- [ ] Multi-channel routing
- [ ] Polyphonic voice management

### Phase 4: Integration
- [ ] Audio-visual coupling examples
- [ ] Audio-driven particle systems
- [ ] Live performance mode (low-latency profile)

## Open Questions

1. **Polyphony**: How to manage voice allocation/stealing?
2. **MIDI**: Native MIDI I/O or file-only?
3. **Modular Routing**: Explicit patch-cord syntax vs. implicit dataflow?
4. **Live Coding**: Hot-reload semantics for audio (click-free transitions)?
