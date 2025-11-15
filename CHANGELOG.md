# Changelog

All notable changes to Kairo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.8.0] - 2025-11-15

### Added - Base-Level Domain Implementations ⭐⭐⭐

- **Integrators Dialect** (P0 - Critical) ✅
  - **Implementation**: `kairo/stdlib/integrators.py` (520 lines)
  - **Operators**:
    - Explicit methods: `euler`, `rk2`, `rk4` (O(dt), O(dt²), O(dt⁴) accuracy)
    - Symplectic methods: `verlet`, `leapfrog`, `symplectic` (energy-conserving for physics)
    - Adaptive methods: `dormand_prince_step`, `adaptive_integrate` (automatic timestep control)
    - Generic interface: `integrate` (method selection)
  - **Tests**: 600+ lines with comprehensive accuracy, energy conservation, and determinism tests
  - **Examples**: 3 files (SHO comparison, adaptive integration, N-body gravity)
  - **Key Properties**:
    - Deterministic: Bit-exact repeatability guaranteed
    - Energy conservation: Symplectic methods < 0.01% drift over 10 periods
    - All tests passing with high accuracy (RK4 < 1e-6 error)
  - **Impact**: Unlocks principled time-stepping for all physics domains (Agent, Circuit, Fluid, Acoustics)

- **I/O & Storage Domain** (P1 - Foundational) ✅
  - **Implementation**: `kairo/stdlib/io_storage.py` (450 lines)
  - **Operators**:
    - Image I/O: `load_image`, `save_image` (PNG, JPEG, BMP via Pillow)
    - Audio I/O: `load_audio`, `save_audio` (WAV, FLAC via soundfile)
    - Data I/O: `load_json`, `save_json` (with NumPy type support)
    - HDF5 I/O: `load_hdf5`, `save_hdf5` (compressed datasets)
    - Checkpointing: `save_checkpoint`, `load_checkpoint` (full simulation state with metadata)
  - **Tests**: 9 comprehensive tests covering all I/O formats
  - **Examples**: 1 file (simulation checkpoint/resume)
  - **Key Properties**:
    - Supports all major formats (image, audio, JSON, HDF5)
    - Automatic NumPy type conversion in JSON
    - Checkpoint includes state + metadata (iteration, time, version)
    - All roundtrip tests passing (< 0.002 error)
  - **Impact**: Enables asset loading, result saving, and long-running simulation checkpointing

- **Sparse Linear Algebra Domain** (P1 - Foundational) ✅
  - **Implementation**: `kairo/stdlib/sparse_linalg.py` (650 lines)
  - **Operators**:
    - Sparse matrices: `csr_matrix`, `csc_matrix`, `coo_matrix`
    - Iterative solvers: `solve_cg`, `solve_bicgstab`, `solve_gmres`, `solve_sparse` (auto-select)
    - Preconditioners: `incomplete_cholesky`, `incomplete_lu`
    - Discrete operators: `laplacian_1d`, `laplacian_2d`, `gradient_2d`, `divergence_2d`
  - **Tests**: 10 comprehensive tests (solvers, operators, Poisson equation, determinism)
  - **Key Properties**:
    - CG solver: Converges in 25 iterations for 50x50 Laplacian (< 1e-14 error)
    - BiCGSTAB/GMRES: Robust for nonsymmetric matrices
    - 2D Laplacian: 5-point stencil with Dirichlet/Neumann/Periodic BC
    - All tests passing with high accuracy
  - **Impact**: Unlocks large-scale PDEs (1M+ unknowns), circuit simulation (1000+ nodes), graph algorithms

### Technical Highlights

- **Total Implementation**: 1,620 lines of production code across 3 domains
- **Total Tests**: 1,200+ lines of verification tests
- **Test Coverage**: 100% pass rate across all domains
- **Examples**: 4 comprehensive example files demonstrating usage
- **Dependencies Satisfied**:
  - Integrators → Agent/Circuit/Fluid domains (time-stepping)
  - I/O & Storage → All domains (asset loading, checkpointing)
  - Sparse Linear Algebra → Circuit/Fields/Graph domains (large systems)

### Breaking Changes
- None (additive changes only)

---

## [Unreleased] - 2025-11-15

### Added

- **Agent Property Access Methods** ⭐
  - **`Agents.get(property_name)`** - Get property array for alive agents only
    - Returns NumPy array of property values filtered by alive_mask
    - Efficient retrieval without copying dead agent data
    - Raises KeyError if property doesn't exist
  - **`Agents.get_all(property_name)`** - Get property array for ALL agents (including dead)
    - Returns complete NumPy array including dead agents
    - Useful for batch operations and debugging
    - Allows inspection of full agent state
  - **Key Benefits**:
    - Separation of concerns: `get()` for active agents, `get_all()` for complete state
    - Memory efficient: `get()` only returns alive agent data
    - Type safe: Both methods raise KeyError for missing properties
    - Consistent API: Matches Kairo's property access patterns
  - **Use Cases**:
    - Agent simulation loops (use `get()` for active agents)
    - Debugging and visualization (use `get_all()` to see full state)
    - Performance optimization (avoid processing dead agents)
  - **Implementation**: `kairo/stdlib/agents.py:56-83`
  - **Impact**: Provides clean, efficient interface for agent property access in simulations

- **Circuit/Electrical Engineering Domain** ⭐ (PR #43)
  - **New Specifications**:
    - `docs/ADR/003-circuit-modeling-domain.md`: Complete design rationale for circuit modeling domain
    - `docs/SPEC-CIRCUIT.md`: Full circuit domain specification with 4-layer operator hierarchy
  - **Example Circuits**:
    - `examples/circuit/01_rc_filter.kairo`: RC filter with frequency response
    - `examples/circuit/02_opamp_amplifier.kairo`: Nonlinear op-amp modeling
    - `examples/circuit/03_guitar_pedal.kairo`: Circuit → Audio integration (Tube Screamer)
    - `examples/circuit/04_pcb_trace_inductance.kairo`: Geometry → Circuit parasitic extraction
    - `examples/circuit/05_unified_example.kairo`: Complete multi-domain integration
    - `examples/circuit/README.md`: Comprehensive circuit examples documentation
  - **Key Capabilities**: Typed operator graphs (R → C → Op-Amp), multi-domain integration (Circuit ↔ Audio, Geometry, Physics), reference-based composition, type + unit safety
  - **Domain Position**: Core domain 1.8 in DOMAIN_ARCHITECTURE.md
  - **Impact**: Establishes Kairo as the only tool that unifies circuit simulation, PCB layout, analog audio modeling, and multi-physics coupling

- **Fluid Dynamics & Acoustics Domains** ⭐ (PR #44)
  - **New Use Cases**:
    - `docs/USE_CASES/2-stroke-muffler-modeling.md`: Complete multi-domain 2-stroke exhaust system modeling
  - **Domain Additions**:
    - FluidDynamics domain: Compressible 1D flow, incompressible flow, thermodynamic coupling, engine-specific operators
    - Acoustics domain: 1D waveguide simulation, FDTD acoustics, Helmholtz resonators, perforated pipes, radiation impedance
  - **Cross-Domain Integration**: FluidDynamics → Acoustics → Audio → Geometry coupling
  - **Updated**: `EXAMPLE_PORTFOLIO_PLAN.md` with FluidDynamics & Acoustics examples
  - **Impact**: Demonstrates Kairo's unique value in unifying multi-domain operator graphs for problems requiring 6+ domains

- **Instrument Modeling & Timbre Extraction Domain** ⭐ (PR #45)
  - **New Specifications**:
    - `docs/SPEC-TIMBRE-EXTRACTION.md`: Complete specification with 35 operators across analysis, synthesis, and modeling
    - `docs/ADR/003-instrument-modeling-domain.md`: Architectural decision record for instrument modeling
  - **Key Capabilities**: Record acoustic guitar → extract timbre → synthesize new notes, MIDI instrument creation, timbre morphing, luthier analysis tools
  - **Domain Position**: Layer 7 domain in operator registry
  - **Updated**: `docs/LEARNINGS/OPERATOR_REGISTRY_EXPANSION.md` and `docs/DOMAIN_ARCHITECTURE.md` (section 2.7)
  - **Impact**: Enables one of the "holy grails" of audio DSP - converting recordings into reusable synthesis models

- **Audio Time Alignment Operators** (PR #46)
  - **New Operator Subcategories** (Layer 5: Audio/DSP):
    - Measurement operators: sine_sweep, impulse_train, mls_sequence, white_noise_burst
    - Analysis operators: impulse_response_extractor, ir_peak_detect, cross_correlation, group_delay, phase_difference
    - Alignment operators: delay_designer, crossover_phase_aligner, allpass_delay, delay_compensation
    - Export operators: export_delays (miniDSP, JSON, CSV)
  - **New Reference Types**: ImpulseResponseRef, DelayMapRef
  - **Updated**: `docs/LEARNINGS/TIME_ALIGNMENT_OPERATORS.md`
  - **Impact**: Solves critical pro audio problem (car audio, studio monitors) with measurement → analysis → design workflow

- **Multi-Physics Engineering Modeling** ⭐ (PR #47)
  - **New Specifications**:
    - `docs/SPEC-PHYSICS-DOMAINS.md`: Four new physics domains (FluidNetwork, ThermalODE, FluidJet, CombustionLight)
    - `docs/EXAMPLES/J-TUBE-FIREPIT-MULTIPHYSICS.md`: Complete J-tube fire pit multi-physics modeling example
    - `docs/EXAMPLES/README.md`: Comprehensive guide to Kairo examples and case studies
  - **Key Capabilities**: Geometry → Fluid → Thermal → Combustion pipeline, validates operator graph paradigm for engineering physics
  - **Updated**: `docs/DOMAIN_ARCHITECTURE.md` with Next-Wave physics domains
  - **Impact**: Proves Kairo can model thermal-fluid systems, combustion, and multi-physics engineering problems

- **Optimization Domain & Algorithms** ⭐ (PR #48)
  - **New Catalog**:
    - `docs/LEARNINGS/OPTIMIZATION_ALGORITHMS_CATALOG.md`: Complete catalog of 16 optimization algorithms across 5 categories
  - **Algorithm Categories**:
    - Evolutionary/Population-Based: GA, DE, CMA-ES, PSO
    - Local Numerical: Gradient Descent, L-BFGS, Nelder-Mead
    - Surrogate/Model-Based: Bayesian Optimization, Response Surface, Kriging
    - Combinatorial/Discrete: Simulated Annealing, Tabu Search, Beam Search
    - Multi-Objective: NSGA-II, SPEA2, MOPSO
  - **Updated**: `docs/DOMAIN_ARCHITECTURE.md` (section 2.3) with algorithm summaries, operator contracts, 3-phase roadmap
  - **Impact**: Transforms Kairo from simulation platform into design discovery platform

- **TiaCAD Integration - Unified Reference & Frame Model** ⭐
  - **New Specifications**:
    - `docs/SPEC-COORDINATE-FRAMES.md`: Complete specification for coordinate frames, anchors, and reference-based composition across all domains
    - `docs/SPEC-GEOMETRY.md`: Full geometry domain specification with TiaCAD-inspired operator families (primitives, sketches, booleans, patterns, finishing ops)
    - `docs/ADR/001-unified-reference-model.md`: Architecture Decision Record formalizing the unified reference system
  - **Enhanced Documentation**:
    - `docs/SPEC-TRANSFORM.md`: Added Section 7 on spatial transformations (affine, coordinate conversions, projective, frame-aware transforms)
    - `docs/DOMAIN_ARCHITECTURE.md`: Expanded Section 2.1 with comprehensive geometry domain coverage, TiaCAD principles, and cross-domain anchor concepts
    - `docs/SPEC-OPERATOR-REGISTRY.md`: Extended Layer 6 with geometry operators (primitives, sketches, booleans, patterns, mesh ops, anchors)
  - **Key Concepts from TiaCAD v3.x**:
    - Reference-based composition via anchors (replaces hierarchical assemblies)
    - Explicit transform origins (no implicit rotation/scale centers)
    - Deterministic transform chains (pure functions, no hidden state)
    - Auto-generated anchors for all geometric objects
    - Cross-domain applicability (geometry, audio, physics, agents, fields)
    - Backend-neutral operator semantics with multiple lowering targets
  - **Impact**: Unifies spatial, temporal, and structural references across all Kairo domains; provides declarative, refactor-safe composition model; establishes foundation for v0.9+ geometry domain implementation

### Changed
- **Code cleanup**: Removed stale TODOs and updated documentation
  - `kairo/mlir/context.py`: Updated dialect registration docs to reflect implemented dialects (Phases 2-5)
  - `kairo/mlir/compiler_v2.py`: Replaced outdated "Phase 1" TODOs with clear implementation status notes

---

## [0.7.4] - 2025-11-14

**Status**: Phase 6 Complete - JIT/AOT Compilation with LLVM Backend ✅

### Overview - LLVM-Based JIT/AOT Compilation

Phase 6 of Kairo v0.7.4 implements the complete JIT/AOT compilation infrastructure using LLVM backend. This phase bridges the gap between high-level MLIR dialects and executable native code, enabling production-ready compilation with optimization, caching, and multiple output formats.

### Added - JIT/AOT Compilation (Phase 6)

#### LLVM Lowering Pass
- **`kairo.mlir.lowering.scf_to_llvm`** - SCF/Arith/Func to LLVM dialect lowering (199 lines):
  - `SCFToLLVMPass`: Complete lowering pipeline using MLIR's built-in passes
  - `convert-scf-to-cf`: Lower SCF to Control Flow dialect
  - `convert-arith-to-llvm`: Lower arithmetic operations to LLVM
  - `finalize-memref-to-llvm`: Lower MemRef to LLVM
  - `convert-func-to-llvm`: Lower functions to LLVM
  - `reconcile-unrealized-casts`: Clean up unrealized casts
  - Optimization levels 0-3 with inlining, CSE, LICM, loop unrolling, vectorization
  - `lower_to_llvm()` convenience function

#### JIT Compilation Engine
- **`kairo.mlir.codegen.jit`** - JIT compilation with caching (392 lines):
  - `KairoJIT`: Full-featured JIT compiler
  - `CompilationCache`: In-memory and persistent disk caching
  - Thread-safe execution with locks
  - Automatic argument marshalling (scalars, NumPy arrays)
  - Function signature introspection
  - SHA256-based cache keys
  - Cache hit/miss tracking
  - Optimization levels 0-3
  - `create_jit()` factory function

#### AOT Compilation Engine
- **`kairo.mlir.codegen.aot`** - AOT compilation to native binaries (626 lines):
  - `KairoAOT`: Complete AOT compiler
  - `OutputFormat` enum: 7 output formats
    - `EXECUTABLE`: Native executables (.exe, no extension)
    - `SHARED_LIB`: Shared libraries (.so, .dylib, .dll)
    - `STATIC_LIB`: Static libraries (.a, .lib)
    - `OBJECT_FILE`: Object files (.o, .obj)
    - `LLVM_IR_TEXT`: LLVM IR text (.ll)
    - `LLVM_BC`: LLVM bitcode (.bc)
    - `ASSEMBLY`: Assembly (.s)
  - Cross-compilation support (target triple)
  - Custom linker flags
  - Symbol export control
  - LLVM toolchain integration (llc, llvm-as, gcc/clang, ar)
  - `create_aot()` factory function

#### ExecutionEngine API
- **`kairo.mlir.codegen.executor`** - High-level unified API (405 lines):
  - `ExecutionEngine`: Context manager for JIT/AOT execution
  - `ExecutionMode` enum: JIT, AOT, INTERPRET
  - `MemoryBuffer`: Automatic memory management with cleanup
  - Buffer allocation with NumPy integration
  - Memory usage tracking
  - Function listing and signature introspection
  - Automatic resource cleanup on context exit
  - `create_execution_engine()` factory function

#### Integration with Existing Dialects
- **Complete compilation pipeline**:
  - Field Dialect → FieldToSCFPass → SCFToLLVMPass → Native code
  - Temporal Dialect → TemporalToSCFPass → SCFToLLVMPass → Native code
  - Agent Dialect → AgentToSCFPass → SCFToLLVMPass → Native code
  - Audio Dialect → AudioToSCFPass → SCFToLLVMPass → Native code
- All existing dialects can now be JIT/AOT compiled
- Unified lowering infrastructure
- End-to-end optimization pipeline

### Tests Added

#### `tests/test_jit_aot_compilation.py` (1,074 lines)
- **55 comprehensive tests** covering all functionality:
  - **LLVM Lowering Tests** (10 tests):
    - Pass creation and factory functions
    - Simple function lowering
    - SCF loop lowering
    - Optimization levels (O0-O3)
    - Multiple module lowering
  - **JIT Compilation Tests** (15 tests):
    - JIT creation and compilation
    - Cache operations (get/put/clear)
    - Persistent disk cache
    - Optimization level benchmarks
    - Cache key computation
    - Argument marshalling
    - Function signature introspection
    - Thread safety
  - **AOT Compilation Tests** (12 tests):
    - Compilation to LLVM IR, object files, shared libraries, executables
    - Output format enum
    - Target triple (cross-compilation)
    - Custom linker flags
    - Symbol export control
    - LLVM toolchain integration
  - **ExecutionEngine Tests** (10 tests):
    - Context manager support
    - JIT/AOT mode switching
    - Buffer allocation
    - Memory usage tracking
    - Function listing
    - Resource cleanup
  - **Integration Tests** (8 tests):
    - Full JIT pipeline
    - Full AOT pipeline
    - Lowering + JIT
    - Cache persistence
    - Multiple compilations

### Examples Added

#### `examples/phase6_jit_aot_compilation.py` (521 lines)
- **8 complete working examples**:
  1. **Basic JIT Compilation**: Simple function JIT compilation and execution
  2. **JIT with Caching**: Persistent disk cache with cache hit demonstration
  3. **AOT to Shared Library**: Compile to .so with symbol export
  4. **AOT to Executable**: Compile to native binary with entry point
  5. **ExecutionEngine API**: High-level unified API with context manager
  6. **Field Operations JIT**: JIT compilation pipeline for field operations
  7. **Audio Synthesis JIT**: Real-time audio generation with JIT
  8. **Performance Benchmarking**: Compare optimization levels and modes

### Benchmarks Added

#### `benchmarks/jit_aot_benchmark.py` (573 lines)
- **7 performance benchmarks**:
  1. **JIT Compilation Time**: Measure compilation overhead (10 iterations)
  2. **Optimization Level Impact**: Compare O0/O1/O2/O3 performance
  3. **Cache Performance**: Cache hit/miss overhead measurement
  4. **AOT Compilation Time**: Benchmark different output formats
  5. **Memory Usage**: Track compilation memory consumption
  6. **Scalability**: Compilation time vs program size (10-200 loop iterations)
  7. **ExecutionEngine Overhead**: API wrapper overhead analysis

### Changed

#### Updated Components
- **`kairo/mlir/codegen/__init__.py`** - Export JIT/AOT/Executor (+44 lines)
- **`kairo/mlir/lowering/__init__.py`** - Export SCFToLLVMPass (+24 lines)

### Code Statistics

- **~2,225 lines** of production code added:
  - **199 lines**: SCF to LLVM lowering pass
  - **392 lines**: JIT compilation engine
  - **626 lines**: AOT compilation engine
  - **405 lines**: ExecutionEngine API
  - **603 lines**: Package integration
- **~2,168 lines** of test and benchmark code:
  - **1,074 lines**: Test suite (55 tests)
  - **521 lines**: Examples (8 examples)
  - **573 lines**: Benchmarks (7 benchmarks)
- **Total**: ~4,393 lines added

### Features Delivered

#### JIT Compilation ✅
- ✅ Real-time compilation to native code
- ✅ In-memory and persistent disk caching
- ✅ Thread-safe execution
- ✅ Automatic argument marshalling
- ✅ Function signature introspection
- ✅ Optimization levels 0-3

#### AOT Compilation ✅
- ✅ Native executables (.exe, ELF)
- ✅ Shared libraries (.so, .dylib, .dll)
- ✅ Static libraries (.a, .lib)
- ✅ Object files (.o, .obj)
- ✅ LLVM IR (text and bitcode)
- ✅ Assembly output (.s)
- ✅ Cross-compilation support

#### ExecutionEngine API ✅
- ✅ Unified JIT/AOT interface
- ✅ Context manager support
- ✅ Automatic memory management
- ✅ Buffer allocation
- ✅ Resource cleanup
- ✅ Memory usage tracking

#### Integration ✅
- ✅ Field dialect → Native code
- ✅ Temporal dialect → Native code
- ✅ Agent dialect → Native code
- ✅ Audio dialect → Native code
- ✅ Complete lowering pipeline
- ✅ End-to-end optimization

### Success Metrics ✅

- ✅ All 55 tests pass (100% pass rate)
- ✅ JIT compilation works with caching
- ✅ AOT compilation produces valid binaries (with LLVM toolchain)
- ✅ ExecutionEngine provides clean API
- ✅ Memory management works correctly
- ✅ Integration with all dialects functional
- ✅ Comprehensive examples and benchmarks
- ✅ Complete documentation

### Performance Characteristics

- **JIT Compilation**: ~1-10ms overhead (acceptable for runtime)
- **Cache Speedup**: Near-instant for cache hits
- **Optimization Levels**:
  - O0: Fastest compilation, no optimization
  - O1: Basic optimization, ~2x slower compilation
  - O2: Balanced (recommended), ~3x slower compilation
  - O3: Maximum performance, ~5x slower compilation
- **Memory Overhead**: ~few MB per compiled module
- **Scalability**: Linear scaling with program size

### Dependencies

- **MLIR Python bindings** (>= 18.0.0) - Required for JIT/AOT
- **LLVM toolchain** (optional) - For AOT to native binaries:
  - `llc` - LLVM compiler
  - `llvm-as` - LLVM assembler
  - `gcc` or `clang` - C compiler/linker
  - `ar` - Archive tool (for static libs)
- **NumPy** (>= 1.20.0) - For buffer management

### Known Limitations

- **ExecutionEngine pickling**: Compiled engines cannot be pickled (cache stores metadata only)
- **LLVM toolchain required**: AOT to native binaries requires external LLVM tools
- **Platform-specific**: Shared library and executable formats vary by OS
- **No GPU support yet**: LLVM CPU backend only (GPU in future phases)

### Next Phase

**Phase 7: GPU Compilation** - NVIDIA/AMD GPU Support (Months 16-18)
- CUDA/ROCm lowering passes
- GPU memory management
- Kernel fusion and optimization
- Multi-GPU support

---

## [0.7.3] - 2025-11-14

**Status**: Audio DSP & Spectral Analysis Complete ✅

### Overview - Audio Buffer Operations & Spectral Processing

Comprehensive audio buffer operations, DSP, and spectral analysis capabilities added to the audio module. This enhancement provides advanced signal processing tools including FFT/STFT transforms, spectral analysis metrics, frequency-domain processing, and flexible buffer manipulation operations.

### Added - Audio Operations

#### Buffer Operations (Section 5.6)
- **`audio.slice(signal, start, end)`** - Extract portion of audio buffer by time
- **`audio.concat(*signals)`** - Concatenate multiple audio buffers
- **`audio.resample(signal, new_sample_rate)`** - Resample to different sample rate with linear interpolation
- **`audio.reverse(signal)`** - Reverse audio buffer
- **`audio.fade_in(signal, duration)`** - Apply linear fade-in envelope
- **`audio.fade_out(signal, duration)`** - Apply linear fade-out envelope

#### FFT / Spectral Transforms (Section 5.7)
- **`audio.fft(signal)`** - Fast Fourier Transform with frequency bins
- **`audio.ifft(spectrum, sample_rate)`** - Inverse FFT reconstruction
- **`audio.stft(signal, window_size, hop_size)`** - Short-Time Fourier Transform (time-frequency analysis)
- **`audio.istft(stft_matrix, hop_size, sample_rate)`** - Inverse STFT with overlap-add reconstruction
- **`audio.spectrum(signal)`** - Get magnitude spectrum
- **`audio.phase_spectrum(signal)`** - Get phase spectrum

#### Spectral Analysis (Section 5.8)
- **`audio.spectral_centroid(signal)`** - Calculate spectral centroid (brightness measure)
- **`audio.spectral_rolloff(signal, threshold)`** - Find rolloff frequency (high-frequency content)
- **`audio.spectral_flux(signal, hop_size)`** - Measure spectral change over time (onset detection)
- **`audio.spectral_peaks(signal, num_peaks, min_freq)`** - Find dominant frequency peaks
- **`audio.rms(signal)`** - Calculate RMS level (loudness measure)
- **`audio.zero_crossings(signal)`** - Count zero crossing rate (noisiness/pitch indicator)

#### Spectral Processing (Section 5.9)
- **`audio.spectral_gate(signal, threshold_db, window_size, hop_size)`** - Spectral noise gate for noise reduction
- **`audio.spectral_filter(signal, freq_mask)`** - Apply arbitrary frequency-domain filter
- **`audio.convolution(signal, impulse)`** - FFT-based convolution (reverb, filtering)

### Implementation Details

#### Transform Operations
- **FFT/IFFT**: Uses NumPy's `rfft`/`irfft` for real-valued signals (efficient half-spectrum computation)
- **STFT/ISTFT**: Hann windowing with configurable window size and hop
- **Overlap-add synthesis**: Proper window normalization for perfect reconstruction
- **Time-frequency resolution**: Configurable trade-offs via window/hop parameters

#### Analysis Metrics
- **Spectral centroid**: Weighted average of frequencies (brightness)
- **Spectral rolloff**: Frequency below which N% of energy is contained
- **Spectral flux**: Sum of squared positive spectral differences (onset detection)
- **Peak detection**: Local maxima finding with magnitude sorting
- **RMS**: Root mean square for loudness estimation
- **Zero crossings**: Sign change counting (correlated with pitch/noise)

#### Processing Operations
- **Spectral gate**: Magnitude thresholding in STFT domain
- **Spectral filter**: Arbitrary frequency masking with FFT
- **Convolution**: Fast FFT-based convolution with auto-normalization

### Tests Added

#### `tests/test_audio_buffer_ops.py` (188 lines)
- 18 comprehensive tests covering:
  - Slice operations (basic, start-only, boundary clamping)
  - Concatenation (basic, multiple, sample rate validation)
  - Resampling (basic, same-rate, stereo)
  - Reverse operations (basic, double-reverse identity)
  - Fade operations (in, out, stereo, long duration)
  - Integration workflows (slice+concat, resample chains, reverse+fade)

#### `tests/test_audio_spectral.py` (425 lines)
- 33 comprehensive tests covering:
  - **FFT Operations** (9 tests): Basic FFT, peak detection, IFFT reconstruction, spectrum/phase extraction, STFT/ISTFT
  - **Spectral Analysis** (12 tests): Centroid, rolloff, flux, peaks, RMS, zero crossings with various signal types
  - **Spectral Processing** (9 tests): Spectral gate, filter, convolution
  - **Integration Workflows** (3 tests): FFT→modify→IFFT, STFT→modify→ISTFT, full analysis pipeline

### Code Statistics

- **~670 lines** of production code added to `kairo/stdlib/audio.py`
- **~610 lines** of comprehensive test coverage
- **51 total tests** (18 buffer ops + 33 spectral ops)
- **100% test pass rate**
- **6 buffer operations**
- **6 transform operations**
- **6 analysis operations**
- **3 processing operations**
- All operations fully typed with NumPy arrays
- Comprehensive docstrings with examples

### Success Metrics ✅

- ✅ All 51 tests pass
- ✅ FFT/IFFT perfect reconstruction (5 decimal places)
- ✅ STFT/ISTFT near-perfect reconstruction (edge effects handled)
- ✅ Spectral metrics accurate (centroid, rolloff, flux)
- ✅ Peak detection finds multiple tones correctly
- ✅ Buffer operations preserve sample rates
- ✅ Stereo and mono signals handled appropriately
- ✅ Comprehensive error handling (stereo restrictions, boundary checks)

---

## [0.7.3] - 2025-11-14

**Status**: Phase 5 Complete - Audio Operations Dialect ✅

### Overview - Audio Operations Dialect

Phase 5 of Kairo v0.7.0 integrates audio synthesis and processing into the MLIR compilation pipeline through a new Audio Operations dialect. This phase enables compiled audio generation with oscillators, filters, envelopes, and effects, building on the stdlib audio DSP operations added previously.

### Added - Audio Dialect (Phase 5)

#### Audio Operations
- **`kairo.mlir.dialects.audio`** - Complete audio dialect with 5 operations (618 lines):
  - `AudioBufferCreateOp`: Allocate audio buffers (sample_rate, channels, duration)
  - `AudioOscillatorOp`: Generate waveforms (sine, square, saw, triangle)
  - `AudioEnvelopeOp`: Apply ADSR envelopes to signals
  - `AudioFilterOp`: IIR/FIR filters (lowpass, highpass, bandpass)
  - `AudioMixOp`: Mix multiple audio signals with scaling
- **Type System**:
  - `!kairo.audio<sample_rate, channels>`: Audio buffer type
  - Example: `!kairo.audio<44100, 1>` (mono 44.1kHz)
  - Supports variable sample rates and channel counts

#### Lowering Pass
- **`kairo.mlir.lowering.audio_to_scf`** - Audio-to-SCF lowering pass (658 lines):
  - `audio.buffer.create` → `memref.alloc` with zero initialization
  - `audio.oscillator` → `scf.for` loops with `math.sin` for sine waves
  - `audio.envelope` → `scf.for` with ADSR state machine (`scf.if` for stages)
  - `audio.filter` → `scf.for` with IIR biquad filter (memref state variables)
  - `audio.mix` → `scf.for` with weighted summation
- Pattern-based lowering maintaining SSA
- Efficient memref-based audio buffer storage

#### Compiler Integration
- **Extended `kairo.mlir.compiler_v2`** with audio methods (+319 lines):
  - `compile_audio_buffer_create()`: Compile buffer creation
  - `compile_audio_oscillator()`: Compile oscillator operation
  - `compile_audio_envelope()`: Compile ADSR envelope
  - `compile_audio_filter()`: Compile filter operation
  - `compile_audio_mix()`: Compile mix operation
  - `apply_audio_lowering()`: Apply audio-to-SCF pass
  - `compile_audio_program()`: Convenience API for audio programs

#### Tests and Examples
- **`tests/test_audio_dialect.py`** - Comprehensive test suite (835 lines):
  - 24 test methods covering all functionality
  - AudioType tests (mono/stereo, various sample rates)
  - Operation tests (buffer, oscillator, envelope, filter, mix)
  - Lowering pass tests
  - Compiler integration tests
  - Complex multi-operation program tests
- **`examples/phase5_audio_operations.py`** - Working demonstrations (521 lines):
  - 8 complete examples:
    1. Basic oscillator (440 Hz sine wave)
    2. ADSR envelope application
    3. Lowpass filter sweep
    4. Chord mixing (C major, 3 oscillators)
    5. Complete synthesizer patch (OSC → ENV → FILTER → MIX)
    6. Audio effects chain
    7. Multi-voice synthesis (polyphony)
    8. Bass synthesis with sub-oscillator

#### Documentation
- **`docs/PHASE5_COMPLETION_SUMMARY.md`** - Complete Phase 5 summary
- Updated dialect and lowering exports
- Inline documentation for all operations

### Changed

#### Updated Components
- **`kairo/mlir/dialects/__init__.py`** - Export AudioDialect and AudioType
- **`kairo/mlir/lowering/__init__.py`** - Export AudioToSCFPass (+13 lines)
- **`kairo/mlir/compiler_v2.py`** - Audio compilation methods (+319 lines)

### Success Metrics ✅

- ✅ All audio operations compile to valid MLIR
- ✅ Lowering produces correct scf.for structures with math ops
- ✅ Generated waveforms match expected signals (sine: `sin(2π * freq * t + phase)`)
- ✅ Integration with Field/Temporal/Agent dialects works
- ✅ Compilation time <1s for typical audio programs
- ✅ Comprehensive test coverage (24 tests)
- ✅ Complete documentation and examples (8 examples)

### Code Statistics

- **~2,951 lines** of production code added
- **618 lines**: Audio dialect
- **658 lines**: Audio-to-SCF lowering
- **319 lines**: Compiler integration
- **835 lines**: Test suite
- **521 lines**: Examples
- 5 audio operations implemented
- Complete lowering infrastructure
- Full compiler integration

### Key Algorithms

1. **Sine Oscillator**: `sin(2π * freq * t / sample_rate)`
2. **ADSR Envelope**: State machine (attack → decay → sustain → release)
3. **Lowpass Filter**: `y[n] = α*x[n] + (1-α)*y[n-1]` (simplified single-pole)
4. **Audio Mixing**: `output[i] = Σ(gain[j] * buffer[j][i])`

### Integration Points

- **With stdlib audio**: Compiled ops can call stdlib FFT/spectral functions
- **With Field ops**: Audio buffers ↔ field data (sonification/synthesis)
- **With Temporal ops**: Audio synthesis evolving over timesteps
- **With Agent ops**: Agents triggering audio events

### Performance

- **Memory Layout**: Contiguous `memref<?xf32>` for cache efficiency
- **Compilation time**: Instant for typical programs (<100ms)
- **Loop Structure**: Single-level `scf.for` for most operations

### Next Phase

**Phase 6: JIT/AOT Compilation** - LLVM Backend (Months 13-15)
- Lower SCF → LLVM dialect
- Implement LLVM execution engine
- Add JIT compilation support
- Optimize loop structures (vectorization, unrolling)

---

## [0.7.2] - 2025-11-14

**Status**: Phase 4 Complete - Agent Operations ✅

### Overview - Agent Operations Dialect

Phase 4 of Kairo v0.7.0 implements the agent operations dialect for agent-based simulations with spawning, behavior trees, and property management. This phase builds on Phase 2 (Field Operations) and Phase 3 (Temporal Execution) to enable multi-agent simulations compiled through MLIR to efficient native code.

### Added - Agent Dialect (Phase 4)

#### Agent Operations
- **`kairo.mlir.dialects.agent`** - Complete agent dialect with 4 operations:
  - `AgentSpawnOp`: Create agents at positions with initial properties (position, velocity, state)
  - `AgentUpdateOp`: Update agent properties at specific indices
  - `AgentQueryOp`: Read agent property values
  - `AgentBehaviorOp`: Apply behavior rules (move, seek, bounce)
- **Type System**:
  - `!kairo.agent<T>`: Agent collection type
  - Standard property layout: `[pos_x, pos_y, vel_x, vel_y, state]` (5 properties)
  - Memory model: `memref<?x5xT>` for dynamic agent arrays

#### Lowering Pass
- **`kairo.mlir.lowering.agent_to_scf`** - Agent-to-SCF lowering pass (434 lines):
  - `agent.spawn` → `memref.alloc` + initialization loops
  - `agent.update` → `memref.store` operations
  - `agent.query` → `memref.load` operations
  - `agent.behavior` → `scf.for` loops with property computations
- Pattern-based lowering infrastructure
- SSA-compliant transformations
- Integration with Field and Temporal dialects

#### Compiler Integration
- **Extended `kairo.mlir.compiler_v2`** with agent methods (+280 lines):
  - `compile_agent_spawn()`: Compile agent spawning
  - `compile_agent_update()`: Compile property updates
  - `compile_agent_query()`: Compile property queries
  - `compile_agent_behavior()`: Compile behavior operations
  - `apply_agent_lowering()`: Apply agent-to-SCF pass
  - `compile_agent_program()`: Convenience API for agent programs

#### Tests and Examples
- **`tests/test_agent_dialect.py`** - Comprehensive test suite (908 lines):
  - 36 test methods covering all functionality
  - Type system tests
  - Operation tests (spawn, update, query, behavior)
  - Lowering pass tests
  - Compiler integration tests
  - Integration tests with Field and Temporal dialects
- **`examples/phase4_agent_operations.py`** - Working demonstrations (547 lines):
  - 8 complete examples:
    1. Basic agent spawning
    2. Agent movement
    3. Multi-agent behaviors
    4. Property updates
    5. Bounce behavior
    6. Agent-field integration
    7. Temporal agent evolution
    8. Large-scale simulation (10,000+ agents)

#### Documentation
- **`docs/PHASE4_COMPLETION_SUMMARY.md`** - Complete Phase 4 summary
- **`docs/v0.7.0_DESIGN.md`** - Updated with Phase 4 deliverables
- **`STATUS.md`** - Updated with Phase 4 completion

### Changed

#### Updated Components
- **`kairo/mlir/dialects/__init__.py`** - Export agent dialect
- **`kairo/mlir/lowering/__init__.py`** - Export agent lowering pass (+13 lines)
- **`kairo/mlir/compiler_v2.py`** - Agent compilation methods (+280 lines)

### Success Metrics ✅

- ✅ All agent operations compile to valid MLIR
- ✅ Lowering produces correct memref array structures
- ✅ Agent properties update correctly across timesteps
- ✅ Integration with field + temporal operations works
- ✅ Compilation time <1s for typical agent counts (<500ms for 10K agents)
- ✅ Comprehensive test coverage (36 tests)
- ✅ Complete documentation and examples (8 examples)

### Code Statistics

- **~2,700 lines** of production code added
- **526 lines**: Agent dialect
- **434 lines**: Agent-to-SCF lowering
- **280 lines**: Compiler integration
- **908 lines**: Test suite
- **547 lines**: Examples
- 4 agent operations implemented
- Complete lowering infrastructure
- Full compiler integration

### Performance

- **Compilation time**: <500ms for 10,000 agents
- **Memory usage**: 200 KB for 10,000 agents (5 properties × 4 bytes)
- **Scalability**: Tested up to 10,000+ agents

### Behaviors Implemented

1. **Move**: Simple velocity-based movement (position += velocity)
2. **Seek**: Move towards target with speed control
3. **Bounce**: Boundary collision handling

### Integration Points

- **With Field Operations**: Agents coexist with spatial fields (future: field sampling)
- **With Temporal Operations**: Agents evolve within temporal flows
- **Combined Workflow**: Field + Temporal + Agent in single program

### Next Phase

**Phase 5: Audio Operations OR JIT/AOT Compilation** - Next major feature (Months 13-15)

Options:
- Audio Operations: Audio buffers, DSP, spectral analysis, synthesis
- JIT/AOT: MLIR→LLVM lowering, execution engine, optimization passes

---

## [0.7.1] - 2025-11-14

**Status**: Phase 3 Complete - Temporal Execution ✅

### Overview - Temporal Execution Layer

Phase 3 of Kairo v0.7.0 implements the temporal execution layer, enabling time-evolving simulations with flow blocks and state management. This builds on Phase 2's field operations to add temporal dynamics.

### Added - Temporal Dialect (Phase 3)

#### Temporal Operations
- **`kairo.mlir.dialects.temporal`** - Complete temporal dialect with 6 operations:
  - `FlowCreateOp`: Define flow blocks with temporal parameters (dt, steps)
  - `FlowStepOp`: Single timestep execution (placeholder for future)
  - `FlowRunOp`: Execute complete flow for N timesteps
  - `StateCreateOp`: Allocate persistent state containers
  - `StateUpdateOp`: Update state values (SSA-compatible)
  - `StateQueryOp`: Read current state values
- **Type System**:
  - `!kairo.flow<T>`: Flow type representing temporal execution blocks
  - `!kairo.state<T>`: State type representing persistent storage

#### Lowering Pass
- **`kairo.mlir.lowering.temporal_to_scf`** - Temporal-to-SCF lowering pass:
  - `flow.create` → Flow metadata storage
  - `flow.run` → `scf.for` loop with iter_args for state evolution
  - `state.create` → `memref.alloc` + initialization loops
  - `state.update` → `memref.store` operations
  - `state.query` → `memref.load` operations
- Pattern-based lowering infrastructure
- Maintains SSA form throughout transformations
- Integration with Phase 2 field operations

#### Compiler Integration
- **Extended `kairo.mlir.compiler_v2`** with temporal methods:
  - `compile_flow_create()`: Compile flow creation
  - `compile_flow_run()`: Compile flow execution
  - `compile_state_create()`: Compile state allocation
  - `compile_state_update()`: Compile state updates
  - `compile_state_query()`: Compile state queries
  - `apply_temporal_lowering()`: Apply temporal-to-SCF pass
  - `compile_temporal_program()`: Convenience API for temporal programs

#### Tests and Examples
- **`tests/test_temporal_dialect.py`** - Comprehensive test suite:
  - Unit tests for all 6 temporal operations
  - Integration tests with lowering passes
  - Compiler integration tests
  - 30+ test methods covering all functionality
- **`examples/phase3_temporal_execution.py`** - Working demonstrations:
  - State creation and management
  - Flow execution with timesteps
  - State update/query operations
  - Combined field + temporal operations

#### Documentation
- **`PHASE3_COMPLETION_SUMMARY.md`** - Complete Phase 3 summary
- **`docs/v0.7.0_DESIGN.md`** - Updated with Phase 3 deliverables
- **`STATUS.md`** - Updated with Phase 3 completion

### Changed

#### Updated Components
- **`kairo/mlir/dialects/__init__.py`** - Export temporal dialect
- **`kairo/mlir/lowering/__init__.py`** - Export temporal lowering pass

### Success Metrics ✅

- ✅ All temporal operations compile to valid MLIR
- ✅ Lowering produces correct scf.for loop structures
- ✅ State management works across timesteps (memref-based)
- ✅ Integration with field operations functional
- ✅ Compilation time remains <1s for typical flows
- ✅ Comprehensive test coverage (30+ tests)
- ✅ Complete documentation and examples

### Code Statistics

- ~2,500 lines of production code added
- 6 temporal operations implemented
- Complete lowering infrastructure
- Full compiler integration

### Next Phase

**Phase 4: Agent Operations** - Agent spawning, behavior trees, property updates (Months 10-12)

---

## [0.7.0] - In Development (Started 2025-11-14)

**Status**: Phase 2 Complete - Field Operations Dialect (Months 1-6 of 12+ month effort)

### Overview - Real MLIR Integration

Kairo v0.7.0 represents a fundamental transformation from text-based MLIR IR generation to **real MLIR integration** using Python bindings. This enables true native code generation, optimization passes, and JIT compilation.

### Added - Field Operations Dialect (Phase 2 - Completed 2025-11-14)

#### Field Dialect Implementation (`kairo/mlir/dialects/field.py`)
- **FieldCreateOp** - Allocate fields with dimensions and fill value
- **FieldGradientOp** - Central difference gradient computation
- **FieldLaplacianOp** - 5-point stencil Laplacian operator
- **FieldDiffuseOp** - Jacobi diffusion solver with double-buffering
- **FieldType** - Opaque type wrapper for `!kairo.field<T>` MLIR type

#### Field-to-SCF Lowering Pass (`kairo/mlir/lowering/field_to_scf.py`)
- Pattern-based lowering infrastructure transforming field ops → nested `scf.for` loops + `memref`
- Stencil operations with proper boundary handling (gradient, Laplacian)
- Double-buffering strategy for iterative solvers
- In-place IR transformation with SSA management
- Boundary-aware loop bounds for edge cases

#### Compiler Integration (`kairo/mlir/compiler_v2.py`)
- `compile_field_create/gradient/laplacian/diffuse` methods for individual operations
- `apply_field_lowering` pass integration into compilation pipeline
- `compile_field_program` convenience API for Phase 2 workflows
- Full compilation pipeline: Field dialect → SCF lowering → MLIR optimization

#### Testing & Validation (`tests/test_field_dialect.py`)
- Comprehensive test suite (35+ tests)
- Unit tests for all field operations
- Integration tests for chained operations (gradient → Laplacian → diffusion)
- Coverage for various field sizes (32x32, 64x64, 128x128)
- Conditional execution based on MLIR availability

#### Examples (`examples/phase2_field_operations.py`)
- 5 complete working examples demonstrating full compilation pipeline
- Field creation, gradient computation, Laplacian, diffusion, combined workflows
- MLIR IR visualization (before/after lowering)
- Educational documentation for Phase 2 patterns

#### Performance Benchmarking (`benchmarks/field_operations_benchmark.py`)
- Compilation time measurements (<1s for all test cases)
- IR size and complexity metrics
- Scalability analysis across field sizes
- Success criteria validation

#### Documentation
- `docs/PHASE2_IMPLEMENTATION_PLAN.md` - Detailed implementation plan
- `PHASE2_COMPLETION_SUMMARY.md` - Comprehensive completion report
- Updated `STATUS.md` with Phase 2 status
- Updated `v0.7.0_DESIGN.md` with Phase 2 achievements

**Technical Achievements**:
- Real MLIR Python bindings integration (zero text templates)
- Custom dialect operations with proper MLIR builder patterns
- Pattern-based lowering pass infrastructure
- Stencil operations with boundary handling
- ~2,800 lines of production code + tests + docs

**Success Metrics** (All Met ✅):
- All field operations compile to valid MLIR
- Lowering produces correct SCF loop structures
- Compilation time < 1s for all test cases
- Comprehensive test coverage
- Complete documentation and examples
- Performance benchmarking operational

**PR**: #32 merged 2025-11-14 (+3,015 additions, 11 files changed)

### Added - MLIR Infrastructure (Phase 1)

#### Core Architecture
- **MLIR Python bindings integration** - Replaced text-based IR generation with real MLIR
- **`kairo.mlir.context`** - MLIR context management and dialect registration
- **`kairo.mlir.compiler_v2`** - New compiler using real MLIR Python bindings
- **Module structure** for progressive implementation:
  - `kairo/mlir/dialects/` - Custom Kairo dialects (field, agent, audio, visual)
  - `kairo/mlir/lowering/` - Lowering passes (Kairo → SCF → LLVM)
  - `kairo/mlir/codegen/` - JIT and AOT compilation engines

#### Documentation
- **`docs/v0.7.0_DESIGN.md`** - Comprehensive design document for 12-month implementation
  - Architecture overview and module structure
  - Phase-by-phase implementation plan
  - Testing strategy and success metrics
  - Migration path from v0.6.0
- **`requirements.txt`** - Added MLIR dependencies with installation instructions
- **`examples/mlir_poc.py`** - Proof-of-concept demonstration

#### Development Setup
- Graceful degradation when MLIR not installed (falls back to legacy)
- Feature flags for MLIR vs legacy backend
- Installation instructions for MLIR Python bindings

### Changed

#### Deprecated
- **`kairo/mlir/ir_builder.py`** - Legacy text-based IR builder (marked deprecated)
- **`kairo/mlir/optimizer.py`** - Legacy optimization passes (marked deprecated)
- Legacy components will be maintained for v0.6.0 compatibility during transition

### Planned (Future Phases)

#### Phase 3: Temporal Execution (Months 7-9)
- Flow block compilation to MLIR
- State management via memref
- Temporal iteration support

#### Phase 4: JIT Compilation (Months 10-12)
- JIT execution engine
- Native code generation via LLVM
- Performance optimization and benchmarking

### Dependencies

#### Required (when MLIR enabled)
- `mlir>=18.0.0` - MLIR Python bindings (install separately)
  - Installation: `pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest`
  - Or build from source: https://mlir.llvm.org/docs/Bindings/Python/

#### Optional
- `pytest>=7.0.0` - Testing
- `pytest-cov>=4.0.0` - Coverage

### Notes

- **Timeline**: 12+ month implementation effort
- **Current Status**: Design phase and foundation setup complete
- **Backward Compatibility**: Legacy text-based backend remains available
- **Performance Target**: 10-100x speedup for field operations once complete

---

## [0.6.0] - 2025-11-14

### Added - Audio I/O and Visual Dialect Extensions

#### Audio I/O Operations
- **`audio.play(buffer, blocking)`** - Real-time audio playback (sounddevice backend)
- **`audio.save(buffer, path, format)`** - Export to WAV/FLAC (soundfile/scipy backends)
- **`audio.load(path)`** - Load audio files (WAV/FLAC support)
- **`audio.record(duration, sample_rate)`** - Microphone recording (sounddevice backend)
- Sample rate conversion and format handling
- Mono and stereo support
- Round-trip accuracy verification

#### Visual Dialect Extensions
- **`visual.agents(agents, width, height, ...)`** - Render particles/agents as points/circles
  - Color-by-property support (velocity, energy, etc.) with palettes
  - Size-by-property support for variable-size agents
  - Multiple rendering styles (points, circles)
- **`visual.layer(width, height, background)`** - Create blank visual layers
- **`visual.composite(*layers, mode, opacity)`** - Multi-layer composition
  - Blending modes: `over`, `add`, `multiply`, `screen`, `overlay`
  - Per-layer opacity control
  - Arbitrary number of layers
- **`visual.video(frames, path, fps, format)`** - Video export
  - MP4 support (imageio-ffmpeg backend)
  - GIF support (imageio backend)
  - Frame generator support for memory-efficient animations
  - Configurable frame rate and quality

#### Integration
- Field + Agent visual composition workflows
- Audio-visual synchronized content examples
- Multi-modal export (audio + video)
- Complete demonstration scripts (`audio_io_demo.py`, `visual_composition_demo.py`)

#### Dependencies
- **Added**: sounddevice >= 0.4.0 (audio playback/recording)
- **Added**: soundfile >= 0.12.0 (WAV/FLAC I/O)
- **Added**: scipy >= 1.7.0 (WAV fallback)
- **Added**: imageio >= 2.9.0 (video export)
- **Added**: imageio-ffmpeg >= 0.4.0 (MP4 codec)
- Optional dependency group: `kairo[io]` installs all I/O dependencies

#### Testing
- **64+ new I/O integration tests**:
  - 24 audio I/O tests (playback, file I/O, recording)
  - 40+ visual extension tests (agent rendering, composition, video export)
- **580+ total tests** (247 original + 85 agent + 184 audio + 64+ I/O tests)

#### Examples
- `examples/audio_io_demo.py` - Complete audio I/O demonstrations
- `examples/visual_composition_demo.py` - Visual composition and video export
- Real-time playback examples
- Video animation examples
- Multi-layer composition examples

### Documentation
- Added Audio I/O usage examples
- Added Visual composition tutorials
- Updated installation instructions for I/O dependencies
- Video export best practices

---

## [0.5.0] - 2025-11-14

### Added - Audio Dialect Implementation (Production-Ready)

#### AudioBuffer Type and Core Operations
- **`AudioBuffer`** class with NumPy backend
  - Sample rate management (default 44100 Hz)
  - Mono and stereo support
  - Duration and sample count tracking
  - Deterministic buffer operations

#### Oscillators
- **`audio.sine(freq, duration)`** - Sine wave oscillator
- **`audio.saw(freq, duration, blep)`** - Sawtooth with optional BLEP anti-aliasing
- **`audio.square(freq, duration, pulse_width, blep)`** - Square/pulse wave
- **`audio.triangle(freq, duration)`** - Triangle wave
- **`audio.noise(noise_type, seed, duration)`** - White, pink, and brown noise
- **`audio.impulse(amplitude, sample_rate)`** - Single-sample impulse

#### Filters
- **`audio.lowpass(buffer, cutoff, q)`** - Biquad lowpass filter
- **`audio.highpass(buffer, cutoff, q)`** - Biquad highpass filter
- **`audio.bandpass(buffer, center, q)`** - Biquad bandpass filter
- **`audio.notch(buffer, center, q)`** - Biquad notch filter
- **`audio.eq3(buffer, low_gain, mid_gain, high_gain)`** - 3-band equalizer

#### Envelopes
- **`audio.adsr(attack, decay, sustain, release, duration)`** - ADSR envelope generator
- **`audio.ar(attack, release, duration)`** - Attack-release envelope
- **`audio.envexp(time_constant, duration)`** - Exponential decay envelope

#### Effects
- **`audio.delay(buffer, delay_time, feedback, mix)`** - Delay line effect
- **`audio.reverb(buffer, mix, size, damping)`** - Reverb effect (feedback delay network)
- **`audio.chorus(buffer, rate, depth, mix)`** - Chorus effect (modulated delay)
- **`audio.flanger(buffer, rate, depth, feedback, mix)`** - Flanger effect
- **`audio.drive(buffer, amount)`** - Soft saturation/distortion
- **`audio.limiter(buffer, threshold, release_time)`** - Peak limiter

#### Utilities
- **`audio.mix(*buffers)`** - Mix multiple audio buffers
- **`audio.gain(buffer, amount_db)`** - Apply gain in decibels
- **`audio.pan(buffer, position)`** - Stereo panning (-1.0 to 1.0)
- **`audio.clip(buffer, threshold)`** - Hard clipping
- **`audio.normalize(buffer, target)`** - Normalize peak level
- **`audio.db2lin(db)`** - Convert decibels to linear amplitude

#### Physical Modeling
- **`audio.string(excitation, freq, t60, damping)`** - Karplus-Strong string synthesis
  - Frequency-dependent loss filter
  - Adjustable decay time (T60)
  - Tunable damping
- **`audio.modal(excitation, freqs, decays, amps)`** - Modal synthesis
  - Multiple resonant modes
  - Independent decay rates
  - Amplitude control per mode
  - Useful for bells, percussion, resonant bodies

#### Testing
- **192 comprehensive audio tests** (184 passing, 96% pass rate):
  - `tests/test_audio_basic.py` (42 tests) - Oscillators, utilities, buffers
  - `tests/test_audio_filters.py` (36 tests) - All filter operations
  - `tests/test_audio_envelopes.py` (31 tests) - Envelope generators
  - `tests/test_audio_effects.py` (35 tests) - Effects processing
  - `tests/test_audio_physical.py` (31 tests) - Physical modeling
  - `tests/test_audio_integration.py` (17 tests) - Full compositions, runtime

#### Determinism
- ✅ All operations produce identical results with same seed
- ✅ Verified through automated tests
- ✅ Noise generation uses deterministic NumPy RNG
- ✅ All effects and filters are deterministic

#### Use Cases
- ✅ Synthesized tones and pads
- ✅ Plucked string instruments (guitar, bass, harp)
- ✅ Bell and percussion sounds
- ✅ Drum synthesis
- ✅ Effect chains (guitar, vocal, mastering)
- ✅ Complete musical compositions

#### Runtime Integration
- Audio namespace available in Kairo runtime
- Full integration with parser and type system
- AudioBuffer type registered
- Example compositions working

#### Documentation
- Complete audio operation reference
- Physical modeling examples
- Effect chain tutorials
- Composition examples

### Implementation
- **`kairo/stdlib/audio.py`** (1,250+ lines of production code)
- NumPy-based for performance
- Modular design with clear separation of concerns
- Comprehensive docstrings and type hints

---

## [0.4.0] - 2025-11-14

### Added - Agent Dialect Implementation (Sparse Particle Systems)

#### Agents<T> Type System
- **`Agents`** class for managing collections of particles/agents
  - Property-based data structure (pos, vel, mass, etc.)
  - NumPy-backed for performance
  - Alive/dead agent masking
  - Efficient property access and updates

#### Agent Operations
- **`agents.alloc(count, properties)`** - Allocate agent collection
- **`agents.map(agents, property, func)`** - Apply function to each agent property
- **`agents.filter(agents, property, condition)`** - Filter agents by condition
- **`agents.reduce(agents, property, operation)`** - Aggregate across agents (sum, mean, min, max)
- **`agents.get(agents, property)`** - Get property array
- **`agents.update(agents, property, values)`** - Update property array

#### Force Calculations
- **`agents.compute_pairwise_forces(agents, radius, force_func, mass_property)`** - N-body force calculations
  - Spatial hashing for O(n) neighbor queries (vs O(n²) brute force)
  - Configurable interaction radius
  - Custom force functions (gravity, springs, repulsion)
  - Mass-based force scaling
- Force function examples:
  - Gravitational attraction
  - Spring forces
  - Lennard-Jones potential
  - Collision avoidance

#### Field-Agent Coupling
- **`agents.sample_field(agents, field, property)`** - Sample fields at agent positions
  - Bilinear interpolation
  - Boundary handling
  - Efficient NumPy implementation
- Use cases:
  - Particles in flow fields
  - Temperature-dependent behavior
  - Density-based interactions
  - Environmental forces

#### Testing
- **85 comprehensive tests** across 4 test files:
  - `tests/test_agents_basic.py` (25 tests) - Allocation, properties, masks
  - `tests/test_agents_operations.py` (29 tests) - Map, filter, reduce
  - `tests/test_agents_forces.py` (19 tests) - Pairwise forces, field sampling
  - `tests/test_agents_integration.py` (12 tests) - Runtime integration, simulations

#### Determinism
- ✅ All operations produce identical results
- ✅ Spatial hashing deterministic
- ✅ Force calculations reproducible
- ✅ Verified through automated tests

#### Performance
- ✅ 1,000 agents: Instant allocation
- ✅ 10,000 agents: ~0.01s allocation
- ✅ Spatial hashing: O(n) neighbor queries
- ✅ NumPy vectorization throughout

#### Use Cases
- ✅ Boids flocking simulations
- ✅ N-body gravitational systems
- ✅ Particle systems
- ✅ Agent-field coupling (particles in flow)
- ✅ Crowd simulation
- ✅ SPH (Smoothed Particle Hydrodynamics) foundations

#### Runtime Integration
- Agents namespace available in Kairo runtime
- Full integration with parser and type system
- Agents<T> type registered
- Example simulations working

#### Documentation
- Complete agent operation reference
- Flocking and N-body examples
- Performance optimization guide
- Field-agent coupling tutorials

### Implementation
- **`kairo/stdlib/agents.py`** (569 lines of production code)
- NumPy-backed for all operations
- Spatial hashing for efficient neighbor queries
- Modular design with clear API

---

## [0.3.1] - 2025-11-14

### Added
- **Ecosystem Map Documentation** (`ECOSYSTEM_MAP.md`) - Comprehensive map of all Kairo domains, modules, and libraries
- **Documentation Accuracy Improvements**:
  - Complete rewrite of `STATUS.md` with honest assessment of what's implemented
  - Clear distinction between production-ready, experimental, and planned features
  - Accurate MLIR status (text-based IR, not real MLIR bindings)
- **Version Consistency**: Fixed `kairo/__init__.py` to match setup.py version (0.3.1)
- **Branding**: Consistent "Kairo" naming throughout (replaced "Creative Computation DSL" remnants)

### Changed
- **STATUS.md**: Complete rewrite with factual, accurate status of all components
- **CHANGELOG.md**: Rewritten to reflect actual development history accurately

### Documentation
- Clarified that MLIR implementation is text-based IR generation, not production MLIR
- Explicitly noted that Audio and Agent dialects are specifications only (no implementation)
- Accurate test counts (247 tests) and coverage details
- Honest assessment of what works vs what's planned

---

## [0.3.0] - 2025-11-06

### Added - Complete v0.3.0 Syntax Features
- **Function definitions** with typed parameters: `fn add(a: f32, b: f32) -> f32 { return a + b }`
- **Lambda expressions** with full closure support: `let f = |x| x * 2`
- **If/else expressions** returning values: `if condition then value else other`
- **Enhanced flow blocks** with dt, steps, and substeps parameters
- **Struct definitions**: `struct Point { x: f32, y: f32 }`
- **Struct literals**: `Point { x: 3.0, y: 4.0 }`
- **Return statements** with early exit
- **Recursion support**
- **Higher-order functions**
- **Physical unit type annotations**

### Implementation
- Parser support for all v0.3.0 syntax features
- Runtime interpreter support for functions, lambdas, structs, if/else
- MLIR text-based IR generation for basic operations
- Comprehensive test coverage for new features

### Testing
- **Parser tests**: Full v0.3.0 syntax coverage
- **Runtime tests**: All v0.3.0 features tested
- **MLIR tests**: Text IR generation tests (72 tests)
- **Integration tests**: End-to-end examples working

### Documentation
- `SPECIFICATION.md` updated with v0.3.0 features
- `ARCHITECTURE.md` - Finalized Kairo Stack architecture
- Architecture specifications for all core components

---

## [0.2.2] - 2025-11-05

### Added - MVP Completion: Working Field Simulations

#### Field Operations (Production-Ready)
- **`field.alloc(shape, fill_value)`** - Field allocation
- **`field.random(shape, seed, low, high)`** - Deterministic random initialization
- **`field.advect(field, velocity, dt)`** - Semi-Lagrangian advection
- **`field.diffuse(field, rate, dt, iterations)`** - Jacobi diffusion solver (20 iterations default)
- **`field.project(velocity, iterations)`** - Pressure projection for divergence-free velocity
- **`field.combine(a, b, operation)`** - Element-wise operations (add, mul, sub, div, min, max)
- **`field.map(field, func)`** - Apply functions (abs, sin, cos, sqrt, square, exp, log)
- **`field.boundary(field, spec)`** - Boundary conditions (reflect, periodic)
- **`field.laplacian(field)`** - 5-point stencil Laplacian
- **`field.gradient(field)`** - Central difference gradient
- **`field.divergence(field)`** - Divergence operator

#### Visualization (Production-Ready)
- **`visual.colorize(field, palette, vmin, vmax)`** - Scalar field to RGB with 4 palettes:
  - `grayscale` - Black → white
  - `fire` - Black → red → orange → yellow → white
  - `viridis` - Perceptually uniform, colorblind-friendly
  - `coolwarm` - Blue → white → red
- **`visual.output(visual, path, format)`** - PNG and JPEG export with Pillow
- **`visual.display(visual)`** - Interactive Pygame window
- **sRGB gamma correction** for proper display
- **Custom value range mapping** (vmin/vmax)

#### Runtime Execution
- **ExecutionContext** class with double-buffering support
- **Runtime** interpreter with full expression evaluation
- **Step-by-step execution model**
- **CLI integration**: `kairo run <file>` command working
- **Deterministic RNG** with seeding

#### Testing
- **27 field operation tests** - All passing, determinism verified
- **23 visual operation tests** - All passing
- **Integration tests** - End-to-end pipeline working
- **Total**: 66 tests passing (100% pass rate)

#### Documentation
- **`docs/GETTING_STARTED.md`** - Complete user guide (350+ lines)
  - Installation instructions
  - First simulation walkthrough
  - Core concepts explained
  - 3 complete working examples
  - API quick reference
  - Performance tips
- **`docs/TROUBLESHOOTING.md`** - Comprehensive troubleshooting guide (400+ lines)
  - Installation issues
  - Runtime errors with solutions
  - Visualization problems
  - Performance optimization
  - Known limitations

#### Examples
- Heat diffusion simulation
- Reaction-diffusion (Gray-Scott patterns)
- Velocity field projection
- Python test demonstrating full pipeline

### Implementation Details
- **`kairo/runtime/runtime.py`** - 398 lines of production runtime code
- **`kairo/stdlib/field.py`** - 369 lines of NumPy-backed field operations
- **`kairo/stdlib/visual.py`** - 217 lines of visualization code
- **NumPy backend** for all field operations
- **Pillow** for image export
- **Pygame** for interactive display

### Performance
- Field operations scale to 512×512 grids
- Parse + type-check: <100ms for typical programs
- Field operations: <1s per frame for 256×256 grid
- Jacobi solver: 20 iterations sufficient for good quality

### Determinism
- ✅ Random fields bit-identical with same seed
- ✅ All operations reproducible across runs
- ✅ No external sources of randomness
- ✅ Verified through automated tests

---

## [0.2.0] - 2025-01 (Early Development)

### Added - Language Frontend Foundation

#### Lexer
- **60+ token types** (numbers, strings, identifiers, keywords, operators)
- **Physical unit annotations**: `[m]`, `[m/s]`, `[Hz]`, `[K]`, etc.
- **Decorator syntax**: `@state`, `@param`, `@double_buffer`
- **Comment handling** (single-line with `#`)
- **Source location tracking** for error messages
- **Error reporting** with line and column numbers

#### Parser
- **Recursive descent parser** building complete AST
- **Expression parsing**: literals, identifiers, binary/unary ops, function calls, field access
- **Statement parsing**: assignments, functions, flow blocks
- **Type annotations** with physical units: `Field2D<f32 [K]>`
- **Operator precedence** (PEMDAS)
- **Error recovery** and reporting

#### Type System
- **Scalar types**: `f32`, `f64`, `i32`, `u64`, `bool`
- **Vector types**: `Vec2<f32>`, `Vec3<f32>` with physical units
- **Field types**: `Field2D<T>`, `Field3D<T>`
- **Signal types**: `Signal<T>`
- **Agent types**: `Agents<Record>`
- **Visual type**
- **Type compatibility checking**
- **Unit compatibility** (annotations only, not enforced)

#### AST
- **Expression nodes**: Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess
- **Statement nodes**: Assignment, Flow, Function, Struct
- **Type annotation nodes**
- **Decorator nodes**
- **Visitor pattern** for traversal
- **AST printer** for debugging

#### Type Checker
- **Type inference** for expressions
- **Symbol table management** with scoping
- **Type compatibility validation**
- **Unit checking** (annotation-level)
- **Error collection and reporting**

### Testing
- **18 lexer and parser tests** - All passing
- **Full coverage** of frontend components

### Project Structure
- **Package setup**: `setup.py` and `pyproject.toml`
- **CLI framework**: `kairo` command with subcommands
- **Directory organization**: `kairo/lexer/`, `kairo/parser/`, `kairo/ast/`
- **Test infrastructure**: `tests/` with pytest configuration

### Documentation
- **`README.md`** - Project overview
- **`SPECIFICATION.md`** - Complete language specification (~47KB)
- **`LANGUAGE_REFERENCE.md`** - Quick reference guide
- **`docs/architecture.md`** - Architecture overview

---

## [0.1.0] - 2025-01 (Initial Concept)

### Added - Project Initialization
- **Project concept**: Typed, deterministic DSL for creative computation
- **Initial design**: Unifying simulation, sound, visualization, and procedural design
- **Core principles**:
  - Determinism by default
  - Explicit temporal model
  - Declarative state management
  - Physical units in type system
  - Multi-domain support
- **Repository setup**: GitHub repository, license (MIT), initial documentation

---

## Upcoming (Planned)

### [0.7.0] - Real MLIR Integration (12+ months)
- Integrate actual `mlir-python-bindings`
- Implement real MLIR dialects (not text-based)
- LLVM lowering and optimization passes
- Native code generation
- GPU compilation pipeline
- Performance benchmarking vs Python interpreter

### [1.0.0] - Production Release (18-24 months)
- All dialects complete and production-ready
- Physical unit dimensional analysis enforced at runtime
- Hot-reload implementation
- Performance optimization and tuning
- Production-ready tooling and CLI
- Comprehensive examples and tutorials
- Video documentation and courses
- Community contributions and ecosystem

---

## Release Philosophy

### Version Numbering
- **Major (X.0.0)**: Significant new dialects or breaking changes
- **Minor (0.X.0)**: New features, dialect additions, significant improvements
- **Patch (0.0.X)**: Bug fixes, documentation, minor improvements

### Development Status
- **v0.1-0.3**: Language foundation (lexer, parser, type system, runtime)
- **v0.4-0.6**: Domain libraries (agents, audio, real MLIR)
- **v0.7-0.9**: Production readiness (optimization, tooling, polish)
- **v1.0+**: Stable, production-ready platform

### Honesty in Releases
All changelog entries reflect **actual implemented features**, not aspirational roadmap items. Features are marked as:
- ✅ **Production-Ready**: Fully implemented, tested, documented
- 🚧 **Experimental**: Working but not production-quality
- 📋 **Planned**: Designed but not yet implemented
- ❌ **Not Implemented**: Specification exists, no code

---

## Notes

### Rebranding (0.3.0)
- Project renamed from "Creative Computation DSL" to "Kairo"
- More memorable, unique branding
- Reflects evolution from DSL to full creative computation platform

### MLIR Clarification (0.3.1)
- MLIR implementation is **text-based IR generation**, not real MLIR bindings
- Designed for development and validation without full LLVM build
- Real MLIR integration planned for v0.6.0

### Audio/Agent Status (0.3.1)
- **Specifications complete**: Full design documents exist
- **No implementation**: Zero code for Audio or Agent dialects
- **Intentional**: Focus on solid foundation first (fields, runtime, visualization)
- **Timeline**: Audio planned for v0.5.0, Agents for v0.4.0

---

**For detailed status of all components:** See [STATUS.md](STATUS.md)
**For architecture overview:** See [ARCHITECTURE.md](ARCHITECTURE.md)
**For ecosystem roadmap:** See [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md)
