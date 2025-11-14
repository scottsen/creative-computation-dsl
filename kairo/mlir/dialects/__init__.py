"""Kairo Custom MLIR Dialects

This package contains custom MLIR dialect definitions for Kairo's
domain-specific operations.

Dialects:
- field: Field operations (gradients, diffusion, advection) - Phase 2 ✅
- temporal: Temporal execution with flow blocks and state - Phase 3 ✅
- agent: Agent-based modeling operations - Phase 4 (TODO)
- audio: Audio synthesis operations - Phase 5 (TODO)
- visual: Visual rendering operations - Phase 6 (TODO)

These dialects are implemented progressively during v0.7.0 development.
"""

# Phase 2: Field dialect
from .field import FieldDialect, FieldType, MLIR_AVAILABLE

# Phase 3: Temporal dialect
from .temporal import TemporalDialect, FlowType, StateType

# TODO: Future phases
# from .agent import KairoAgentDialect
# from .audio import KairoAudioDialect
# from .visual import KairoVisualDialect

__all__ = [
    "FieldDialect",
    "FieldType",
    "TemporalDialect",
    "FlowType",
    "StateType",
    "MLIR_AVAILABLE",
]
