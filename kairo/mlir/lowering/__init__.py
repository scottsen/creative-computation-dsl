"""MLIR Lowering Passes for Kairo

This package contains lowering passes that transform Kairo's high-level
dialects into progressively lower-level representations:

Kairo Dialects → SCF/Arith/Func → LLVM Dialect → LLVM IR → Native Code

Passes:
- FieldToSCFPass: Lower field operations to structured control flow (Phase 2)
- TemporalToSCFPass: Lower temporal operations to SCF loops (Phase 3)
- SCFToLLVMPass: Lower SCF to LLVM dialect (TODO Phase 4)
- OptimizationPasses: MLIR optimization passes (TODO Phase 4)
"""

# Phase 2 passes
from .field_to_scf import FieldToSCFPass, create_field_to_scf_pass, MLIR_AVAILABLE

# Phase 3 passes
from .temporal_to_scf import TemporalToSCFPass, create_temporal_to_scf_pass

# TODO: Phase 4 passes
# from .scf_to_llvm import SCFToLLVMPass
# from .optimization import create_optimization_pipeline

__all__ = [
    "FieldToSCFPass",
    "create_field_to_scf_pass",
    "TemporalToSCFPass",
    "create_temporal_to_scf_pass",
    "MLIR_AVAILABLE",
]
