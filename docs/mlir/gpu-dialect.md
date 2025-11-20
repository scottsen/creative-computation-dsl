# MLIR GPU Dialect Integration

**Version:** 0.12.0 (Phase 6)
**Status:** Implemented ✅
**Last Updated:** 2025-11-20

---

## Overview

The Morphogen GPU dialect provides high-level GPU operations that lower to MLIR's built-in GPU dialect, enabling GPU-accelerated execution of Morphogen programs. This integration leverages the existing 6-dialect compiler infrastructure (field, temporal, agent, audio, visual, transform) and adds GPU parallel execution capabilities.

## Architecture

### Design Principles

The GPU dialect integration follows the [GPU & MLIR Principles](../architecture/gpu-mlir-principles.md) design guidelines:

1. **Express Parallelism Structurally** - Use explicit iteration spaces
2. **Model Memory Hierarchy** - Support global/shared/register memory
3. **Follow Canonical Pipeline** - Tile → Vectorize → GPU-map
4. **Static Shapes** - Prefer compile-time constants
5. **Determinism Profiles** - Support strict/repro/live execution modes

### Compilation Pipeline

```
Morphogen Operations (field, temporal, agent, audio)
    ↓
Custom Dialect Operations
    ↓
SCF Loops (Structured Control Flow)
    ↓
GPU Dialect Operations [NEW]
    ↓
MLIR GPU Dialect (gpu.launch, gpu.thread_id, etc.)
    ↓
NVVM/ROCDL/SPIRV
    ↓
PTX/LLVM/SPIR-V → Native GPU Code
```

## GPU Dialect Types

### 1. `!morphogen.gpu.kernel<T>`

GPU kernel handle with element type.

```mlir
%kernel = morphogen.gpu.launch ... : !morphogen.gpu.kernel<f32>
```

### 2. `!morphogen.gpu.buffer<T>`

GPU global memory buffer.

```mlir
%buffer = morphogen.gpu.alloc(%height, %width) : !morphogen.gpu.buffer<f32>
```

### 3. `!morphogen.gpu.shared<T>`

GPU shared memory buffer (fast, limited size ~48KB per block).

```mlir
%shared = morphogen.gpu.alloc_shared(%size) : !morphogen.gpu.shared<f32>
```

## GPU Dialect Operations

### 1. `morphogen.gpu.launch`

Launch a GPU kernel with specified block and thread configuration.

**Syntax:**
```mlir
%result = morphogen.gpu.launch %blocks_x, %blocks_y, %blocks_z,
                               %threads_x, %threads_y, %threads_z
          (%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>)
```

**Parameters:**
- `blocks`: Grid dimensions [x, y, z]
- `threads`: Block dimensions [x, y, z]
- `args`: Kernel arguments

**Lowering Target:** `gpu.launch_func`

### 2. `morphogen.gpu.parallel`

Express data-parallel computation.

**Syntax:**
```mlir
%result = morphogen.gpu.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1)
          in (%input) : memref<?x?xf32> {
    ... computation ...
    morphogen.gpu.yield %value
} -> memref<?x?xf32>
```

**Lowering Target:** `gpu.launch` with thread/block mapping

### 3. `morphogen.gpu.alloc`

Allocate GPU global memory.

**Syntax:**
```mlir
%buffer = morphogen.gpu.alloc(%height, %width) : !morphogen.gpu.buffer<f32>
```

**Lowering Target:** `gpu.alloc` or `memref.alloc` with GPU address space

### 4. `morphogen.gpu.alloc_shared`

Allocate GPU shared memory (fast, limited size).

**Syntax:**
```mlir
%shared = morphogen.gpu.alloc_shared(%size) : !morphogen.gpu.shared<f32>
```

**Lowering Target:** `gpu.dynamic_shared_memory`

### 5. `morphogen.gpu.sync`

Thread synchronization barrier.

**Syntax:**
```mlir
morphogen.gpu.sync
```

**Purpose:** Ensures all threads in a block reach this point before continuing. Required when using shared memory to avoid race conditions.

**Lowering Target:** `gpu.barrier`

### 6. `morphogen.gpu.thread_id`

Get current thread/block index.

**Syntax:**
```mlir
%thread_x = morphogen.gpu.thread_id "x"          // Thread ID in X dimension
%thread_y = morphogen.gpu.thread_id "y"          // Thread ID in Y dimension
%block_x = morphogen.gpu.thread_id "block_x"     // Block ID in X dimension
```

**Dimensions:** `"x"`, `"y"`, `"z"` for threads, `"block_x"`, `"block_y"`, `"block_z"` for blocks

**Lowering Target:** `gpu.thread_id`, `gpu.block_id`

## Lowering Passes

### 1. SCF-to-GPU Pass

Transforms SCF loops into GPU parallel operations.

**Input (SCF):**
```mlir
scf.for %i = %c0 to %h step %c1 {
  scf.for %j = %c0 to %w step %c1 {
    %val = memref.load %field[%i, %j]
    %result = math.sin %val
    memref.store %result, %out[%i, %j]
  }
}
```

**Output (GPU-annotated):**
```mlir
scf.for %i = %c0 to %h step %c1 {
  gpu_parallelizable = true
  gpu_dimension = "y"
  block_size_y = 16

  scf.for %j = %c0 to %w step %c1 {
    gpu_parallelizable = true
    gpu_dimension = "x"
    block_size_x = 16

    %val = memref.load %field[%i, %j]
    %result = math.sin %val
    memref.store %result, %out[%i, %j]
  }
}
```

### 2. Field-to-GPU Pass

Combined pass that directly transforms field operations to GPU execution.

**Steps:**
1. Field operations → SCF loops (existing `FieldToSCFPass`)
2. SCF loops → GPU parallel (new `SCFToGPUPass`)

## Usage Examples

### Python API

#### 1. Compile Field Operations with GPU Acceleration

```python
from morphogen.mlir.context import MorphogenMLIRContext
from morphogen.mlir.compiler_v2 import MLIRCompilerV2

# Create context and compiler
ctx = MorphogenMLIRContext()
compiler = MLIRCompilerV2(ctx)

# Define field operations
operations = [
    {"op": "create", "args": {"width": 256, "height": 256, "fill": 0.0}},
    {"op": "gradient", "args": {"field": "field0"}},
]

# Compile with GPU acceleration
module = compiler.compile_field_program_gpu(
    operations,
    module_name="field_gpu_program",
    block_size=[16, 16, 1]  # 2D block: 16x16 threads
)

# Verify and print MLIR
module.operation.verify()
print(module)
```

#### 2. Apply GPU Lowering to Existing Module

```python
# Compile regular field program
module = compiler.compile_field_program(operations)

# Apply GPU lowering
compiler.apply_gpu_lowering(module, block_size=[32, 32, 1])
```

#### 3. Combined Field-to-GPU Lowering

```python
# Apply combined pass
compiler.apply_field_to_gpu_lowering(module, block_size=[16, 16, 1])
```

### Block Size Configuration

Block size determines how many threads execute in parallel per GPU block.

**Common Configurations:**

| Use Case | Block Size | Description |
|----------|------------|-------------|
| 1D Processing | `[256, 1, 1]` | Audio streams, 1D signals |
| 2D Fields | `[16, 16, 1]` | Image processing, spatial fields |
| 2D Large | `[32, 32, 1]` | Large field operations |
| 3D Volumes | `[8, 8, 8]` | Volumetric data |

**Guidelines:**
- Total threads per block ≤ 1024 (GPU hardware limit)
- Prefer powers of 2 for optimal warp utilization
- 2D default: `[16, 16, 1]` = 256 threads (good balance)
- 1D default: `[256, 1, 1]` = 256 threads

## Integration with Existing Dialects

The GPU dialect integrates seamlessly with all existing Morphogen dialects:

### Field Dialect + GPU

```python
# Create field, compute gradient on GPU
operations = [
    {"op": "create", "args": {"width": 512, "height": 512, "fill": 1.0}},
    {"op": "gradient", "args": {"field": "field0"}},
]
module = compiler.compile_field_program_gpu(operations, block_size=[16, 16, 1])
```

### Temporal Dialect + GPU

Future: GPU-accelerated time-stepping simulations.

```python
# Time-evolving field simulation on GPU
# Coming in future phase
```

### Agent Dialect + GPU

Future: GPU-accelerated agent-based modeling.

```python
# Parallel agent updates on GPU
# Coming in future phase
```

### Audio Dialect + GPU

Future: GPU-accelerated audio synthesis.

```python
# Parallel oscillator banks on GPU
# Coming in future phase
```

## Determinism Profiles

The GPU dialect supports Morphogen's determinism profiles:

### Strict Profile
```python
# Fixed tile sizes, deterministic execution
module = compiler.compile_field_program_gpu(
    operations,
    block_size=[256, 1, 1],  # Fixed block size
    # Future: deterministic reduction modes
)
```

### Repro Profile
```python
# Reproducible within floating-point precision
# Auto-tuned block sizes allowed
module = compiler.compile_field_program_gpu(operations)
```

### Live Profile
```python
# Low-latency, potentially non-deterministic optimizations
# Future: fast atomics, non-deterministic reductions
```

## Testing

Tests are provided in `tests/mlir/test_gpu_dialect.py`:

```bash
pytest tests/mlir/test_gpu_dialect.py -v
```

**Test Coverage:**
- GPU dialect type creation
- GPU operation creation
- SCF-to-GPU lowering pass
- Field-to-GPU compilation
- Block size configuration
- Module verification

## Performance Considerations

### Memory Coalescing

GPU performance depends heavily on memory access patterns:

**Good (Coalesced):**
```mlir
// Adjacent threads access adjacent memory
%val = memref.load %field[%i, %thread_id_x]
```

**Bad (Strided):**
```mlir
// Adjacent threads access strided memory
%val = memref.load %field[%thread_id_x, %i]
```

### Shared Memory Optimization

Use shared memory for data reuse (future enhancement):

```mlir
// Load tile into shared memory
%shared = morphogen.gpu.alloc_shared(%tile_size)
// Synchronize
morphogen.gpu.sync
// Compute from shared memory
%val = memref.load %shared[%local_idx]
```

### Block Size Tuning

- **Small blocks** (e.g., 64 threads): Good for memory-bound ops
- **Large blocks** (e.g., 256-512 threads): Good for compute-bound ops
- **2D balanced** (e.g., 16×16): Good default for fields

## Future Enhancements

### Phase 7 Extensions

1. **Direct GPU Dialect Lowering**
   - Lower to actual `gpu.launch_func` (currently uses annotations)
   - Generate GPU kernels in separate modules

2. **Shared Memory Optimization**
   - Auto-detect tile reuse patterns
   - Insert shared memory allocations
   - Add synchronization barriers

3. **Multi-Dialect GPU Support**
   - Temporal operations on GPU
   - Agent-based modeling on GPU
   - Audio synthesis on GPU

4. **Advanced Optimizations**
   - Warp-level primitives
   - Cooperative groups
   - Tensor core operations

5. **Backend Integration**
   - CUDA backend (NVVM lowering)
   - ROCm backend (ROCDL lowering)
   - Vulkan/Metal backend (SPIRV lowering)

## References

- [GPU & MLIR Principles](../architecture/gpu-mlir-principles.md)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [Morphogen Architecture](../ARCHITECTURE.md)
- [Field Dialect Specification](field-dialect.md)

## API Reference

### Python Classes

#### `GPUDialect`

Namespace for GPU operations.

**Methods:**
- `launch(blocks, threads, args, element_type, loc, ip)` - Launch GPU kernel
- `parallel(lower_bounds, upper_bounds, inputs, result_type, loc, ip)` - Data-parallel op
- `alloc(dims, element_type, loc, ip)` - Allocate GPU global memory
- `alloc_shared(size, element_type, loc, ip)` - Allocate GPU shared memory
- `sync(loc, ip)` - Thread synchronization barrier
- `thread_id(dimension, loc, ip)` - Get thread/block index

#### `SCFToGPUPass`

Lowering pass: SCF loops → GPU parallel execution.

**Constructor:**
```python
SCFToGPUPass(context, block_size=None, use_shared_memory=False)
```

**Parameters:**
- `context`: MorphogenMLIRContext
- `block_size`: Threads per block [x, y, z] (default: [256, 1, 1])
- `use_shared_memory`: Enable shared memory optimization

**Methods:**
- `run(module)` - Apply pass to module

#### `FieldToGPUPass`

Combined pass: Field operations → GPU execution.

**Constructor:**
```python
FieldToGPUPass(context, block_size=None)
```

**Parameters:**
- `context`: MorphogenMLIRContext
- `block_size`: GPU block size [x, y, z] (default: [16, 16, 1])

**Methods:**
- `run(module)` - Apply pass to module

#### `MLIRCompilerV2` (Extended)

Compiler with GPU support.

**New Methods:**
- `apply_gpu_lowering(module, block_size=None)` - Apply SCF-to-GPU pass
- `apply_field_to_gpu_lowering(module, block_size=None)` - Apply Field-to-GPU pass
- `compile_field_program_gpu(operations, module_name, block_size=None)` - Compile fields with GPU

## Changelog

### v0.12.0 (2025-11-20)
- ✅ Initial GPU dialect implementation
- ✅ GPU types: kernel, buffer, shared
- ✅ GPU operations: launch, parallel, alloc, alloc_shared, sync, thread_id
- ✅ SCF-to-GPU lowering pass
- ✅ Field-to-GPU combined pass
- ✅ Compiler integration with `compile_field_program_gpu()`
- ✅ Test suite for GPU dialect
- ✅ Documentation and examples

---

*This document is part of the Morphogen MLIR compiler documentation. For questions or contributions, see the main [ARCHITECTURE.md](../ARCHITECTURE.md).*
