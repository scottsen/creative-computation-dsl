# SPEC: Transform Dialect

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-11

---

## Overview

The Transform Dialect makes **domain changes** a first-class operation in Kairo. Transforms are not special-cased utilities; they are core grammatical elements with strict semantics, deterministic behavior, and profile-driven tuning.

**Design Principle:** FFT is not special — it's one instance of `transform.to(domain="frequency")`. All transforms follow the same pattern.

---

## Core Operations

### `transform.to(x, domain, method, **attrs) -> Stream`

Convert stream `x` to a different domain representation.

**Parameters:**
- `x`: Source stream (any `Stream<T,D,R>`)
- `domain`: Target domain name (string)
- `method`: Transform method (domain-specific)
- `**attrs`: Transform-specific attributes (window, overlap, family, etc.)

**Returns:** `Stream<T',D',R'>` in target domain

**Example:**
```kairo
let spec = transform.to(signal, domain="frequency", method="fft", window="hann")
```

---

### `transform.from(x, domain, method, **attrs) -> Stream`

Convert stream `x` back from a domain representation.

**Parameters:**
- `x`: Source stream in transformed domain
- `domain`: Source domain to convert from
- `method`: Inverse transform method
- `**attrs`: Transform-specific attributes

**Returns:** `Stream<T,D,R>` in original domain

**Example:**
```kairo
let signal = transform.from(spectrum, domain="frequency", method="ifft")
```

---

### `transform.reparam(x, mapping) -> Stream`

Apply a coordinate transformation without changing domain.

**Parameters:**
- `x`: Source stream
- `mapping`: Coordinate mapping function or matrix

**Returns:** `Stream<T,D,R>` with transformed coordinates

**Example:**
```kairo
# Mel-frequency warping
let mel_spec = transform.reparam(spectrum, mapping=mel_scale(n_mels=128))
```

---

## Domain Pairs & Methods

### 1. Time ↔ Frequency

**Forward Transforms:**

#### FFT (Fast Fourier Transform)
```kairo
transform.to(sig, domain="frequency", method="fft",
             window="hann",      # Window function
             nfft=null,          # FFT size (default: signal length)
             center=true,        # Center window
             norm="ortho")       # Normalization mode
```

**Attributes:**
- `window`: `"hann"`, `"hamming"`, `"blackman"`, `"kaiser"`, `"rectangular"`
- `nfft`: FFT size (power of 2, >= signal length)
- `center`: Center windowing (bool)
- `norm`: `"ortho"` (√N), `"forward"` (1), `"backward"` (1/N)

**Returns:** `Stream<Complex<f32>, 1D, audio>` with frequency bins

---

#### STFT (Short-Time Fourier Transform)
```kairo
transform.to(sig, domain="frequency", method="stft",
             window="hann",
             n_fft=2048,
             hop_length=512,
             center=true,
             norm="ortho")
```

**Attributes:**
- `window`: Window function
- `n_fft`: FFT size per frame
- `hop_length`: Samples between frames
- `center`: Pad signal for centered windows
- `norm`: Normalization mode

**Returns:** `Stream<Complex<f32>, 2D, audio>` (time × frequency)

---

**Inverse Transforms:**

```kairo
# IFFT
transform.from(spec, domain="frequency", method="ifft",
               length=null,  # Output length (default: infer from spectrum)
               norm="ortho")

# ISTFT (overlap-add reconstruction)
transform.from(stft, domain="frequency", method="istft",
               hop_length=512,
               window="hann",
               center=true,
               length=null)
```

---

### 2. Time ↔ Cepstral

#### DCT (Discrete Cosine Transform)
```kairo
transform.to(sig, domain="cepstral", method="dct",
             type=2,      # DCT type (1-4)
             norm="ortho")
```

**Use cases:** Compression, MFCC computation, cepstral analysis

**Inverse:**
```kairo
transform.from(ceps, domain="cepstral", method="idct", type=2, norm="ortho")
```

---

### 3. Time ↔ Wavelet

#### Wavelet Transform
```kairo
transform.to(sig, domain="wavelet", method="cwt",
             wavelet="morlet",   # Wavelet family
             scales=[1..128],    # Scale values
             sampling_period=1.0)
```

**Wavelet families:** `"morlet"`, `"mexican_hat"`, `"paul"`, `"dog"` (derivative of Gaussian)

**Returns:** `Stream<Complex<f32>, 2D, audio>` (scale × time)

**Inverse:**
```kairo
transform.from(cwt, domain="wavelet", method="icwt", wavelet="morlet")
```

---

### 4. Space ↔ k-space (Spatial Frequency)

For 2D/3D fields (PDEs, images, volumes):

```kairo
# 2D Fourier transform (spatial → k-space)
let k_field = transform.to(field, domain="k-space", method="fft2d",
                            norm="ortho")

# Apply filter in k-space (e.g., low-pass)
let filtered_k = k_field * gaussian_kernel(sigma=5.0)

# Transform back to spatial domain
let filtered = transform.from(filtered_k, domain="k-space", method="ifft2d")
```

**Use cases:** PDE spectral methods, image filtering, diffusion

---

### 5. Linear ↔ Perceptual

#### Mel Scale (Frequency Warping)
```kairo
# Frequency → Mel frequency
let mel_spec = transform.reparam(spectrum, mapping=mel_scale(
    n_mels=128,
    fmin=0Hz,
    fmax=8000Hz
))

# Mel frequency → Frequency
let lin_spec = transform.reparam(mel_spec, mapping=inverse_mel_scale())
```

**Use cases:** Perceptual audio features, voice processing

---

### 6. Graph ↔ Spectral

For graph/network data:

```kairo
# Graph Laplacian eigenbasis
let spectral = transform.to(graph, domain="spectral", method="laplacian",
                             k=50)  # Number of eigenvectors

# Graph signal filtering
let filtered = spectral * spectral_filter

# Back to graph domain
let smooth = transform.from(filtered, domain="spectral", method="inverse_laplacian")
```

**Use cases:** Graph signal processing, network analysis, smoothing

---

## Determinism & Profiles

### Strict Profile
- **Bit-exact** FFT (aligned to reference implementations)
- Fixed normalization
- Deterministic phase handling
- Golden test vectors included

### Repro Profile
- **Deterministic within floating-point precision**
- Vendor FFT libraries allowed (FFTW, MKL)
- Consistent windowing/normalization

### Live Profile
- **Lowest latency**
- Allows approximations (e.g., shorter FFT, lower overlap)
- Replayable but not bit-exact

---

## Normalization Modes

All transforms support explicit normalization control:

- `"ortho"`: Orthonormal (forward and inverse both √N)
- `"forward"`: Forward scaled by 1, inverse by 1/N
- `"backward"`: Forward scaled by 1/N, inverse by 1

**Default:** `"ortho"` (symmetric, energy-preserving)

---

## Window Functions

Standard window functions for all time-frequency transforms:

| Window | Formula | Use Case |
|--------|---------|----------|
| `hann` | Cosine-squared | General purpose, good sidelobe suppression |
| `hamming` | Raised cosine | Slightly better frequency resolution |
| `blackman` | Three-term cosine | Excellent sidelobe rejection |
| `kaiser` | Bessel-based, tunable β | Adjustable tradeoff (β parameter) |
| `rectangular` | No tapering | Maximum frequency resolution (high leakage) |

**Profile influence:**
- Strict: Exact window coefficients (reference implementation)
- Repro: Vendor-optimized windows (consistent results)
- Live: Fast approximations allowed

---

## Error Handling

### Type Errors
```kairo
# ERROR: Cannot FFT a 2D field
let spec = transform.to(field2d, domain="frequency", method="fft")
# → Use method="fft2d" for 2D data
```

### Domain Mismatches
```kairo
# ERROR: Inverse domain must match forward domain
let spec = transform.to(sig, domain="frequency", method="fft")
let back = transform.from(spec, domain="wavelet", method="icwt")
# → Domain mismatch: expected "frequency"
```

### Attribute Validation
```kairo
# ERROR: hop_length must divide n_fft evenly for perfect reconstruction
transform.to(sig, domain="frequency", method="stft", n_fft=2048, hop_length=513)
```

---

## Implementation Notes

### Phase 1 (v0.4.0)
- ✅ FFT/IFFT (1D, time↔frequency)
- ✅ STFT/ISTFT (spectrogram)
- ✅ Window functions (hann, hamming, blackman)
- ✅ Profile-driven normalization

### Phase 2 (v0.5.0)
- DCT/IDCT (cepstral)
- Wavelet transforms (CWT/ICWT)
- Mel-scale warping

### Phase 3 (v0.6.0)
- FFT2D/IFFT2D (space↔k-space)
- Graph spectral transforms
- Vendor FFT provider integration (FFTW, MKL, cuFFT)

---

## Examples

### Example 1: Spectral Filtering
```kairo
scene SpectralFilter {
  let sig = sine(440Hz) + sine(880Hz) + noise(seed=42) * 0.1

  # Transform to frequency domain
  let spec = transform.to(sig, domain="frequency", method="fft", window="hann")

  # Apply filter (zero out high frequencies)
  let filtered_spec = spec * lowpass_mask(cutoff=1000Hz)

  # Transform back
  let clean = transform.from(filtered_spec, domain="frequency", method="ifft")

  out mono = clean
}
```

### Example 2: STFT-based Processing
```kairo
scene VocoderEffect {
  let voice = input_mono()

  # Compute STFT
  let stft = transform.to(voice, domain="frequency", method="stft",
                          n_fft=2048, hop_length=512)

  # Spectral manipulation (phase vocoder stretch)
  let stretched = time_stretch(stft, factor=1.5)

  # Reconstruct
  let output = transform.from(stretched, domain="frequency", method="istft",
                              hop_length=512)

  out mono = output
}
```

### Example 3: Mel-Frequency Features
```kairo
scene MelFeatures {
  let audio = input_mono()

  # Compute spectrogram
  let spec = transform.to(audio, domain="frequency", method="stft", n_fft=2048)

  # Convert to Mel scale
  let mel = transform.reparam(spec, mapping=mel_scale(n_mels=128, fmax=8000Hz))

  # Log magnitude
  let log_mel = log(abs(mel) + 1e-8)

  # Visualize
  out visual = colorize(log_mel, palette="viridis")
}
```

---

## Summary

The Transform Dialect provides:

✅ **Uniform grammar** for all domain changes
✅ **First-class transforms** (not library calls)
✅ **Profile-driven determinism** (strict/repro/live)
✅ **Composable operations** (chain transforms, mix domains)
✅ **Extensible** (new domains/methods via registry)

Transforms are the bridge between Kairo's multi-domain vision and practical computation.
