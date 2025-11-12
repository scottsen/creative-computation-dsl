# Kairo Architecture Analysis: Three-Layer Proposal Validation

**Date:** 2025-11-12
**Purpose:** Validate proposed three-layer architecture against actual codebase
**Status:** Analysis Complete

---

## Executive Summary

This document analyzes a proposed three-layer architecture (RiffStack → Kairo.Audio → Kairo Core) against the actual Kairo codebase. **Key finding:** The proposed architecture does not match current reality. Kairo is currently a unified DSL project, and RiffStack is mentioned as a separate sibling project with no code in this repository.

**Verdict:** The ChatGPT proposal contains interesting architectural ideas but was based on incomplete/incorrect understanding of the project. Some concepts are worth considering for future evolution, but significant clarification and planning would be needed.

---

## Current Reality: What Actually Exists

### Kairo v0.3.1 - Unified Creative Computation DSL

**Status:** Active, production-ready MLIR pipeline

**Components:**
```
kairo/
├── lexer/          ✅ Complete tokenization
├── parser/         ✅ Complete AST generation
├── ast/            ✅ Typed AST with visitor pattern
├── types/          ✅ Type system with physical units
├── mlir/           ✅ 100% complete compilation pipeline
│   ├── compiler.py     (Phases 1-4: operations, control flow, temporal, lambdas)
│   ├── optimizer.py    (Phase 5: constant folding, DCE, simplification)
│   └── ir_builder.py   (IR construction utilities)
├── runtime/        ✅ Flow scheduler, state management, RNG
├── stdlib/         ✅ Field, agent, signal, visual operations
└── cli.py          ✅ kairo run/check/parse/mlir commands
```

**Test Coverage:**
- 232 total tests passing
- 72 MLIR-specific tests (100% pass rate)
- Integration tests for all language features

**Architecture:**
- **Single unified system** - not layered
- Direct compilation: Source → Parser → AST → MLIR → (future: LLVM)
- Four dialects as features, not layers: Field, Agent, Signal, Visual

### RiffStack - Status Unknown

**Mentioned in README:**
> "RiffStack - Audio-focused sibling project
> While Kairo is a multi-domain creative computation platform, RiffStack focuses specifically on audio synthesis and live performance."

**Reality Check:**
- ❌ No RiffStack code in this repository
- ❌ No import/export between RiffStack and Kairo
- ❌ No shared operator registry
- ❌ Mentioned as separate GitHub project, not a layer

**Conclusion:** RiffStack is either:
1. A separate project that hasn't been started yet
2. A separate project in a different repository
3. A concept/proposal, not implemented

### Version Confusion

There's a discrepancy in the documentation:
- **README.md:** Claims v0.3.1 (updated 2025-11-06)
- **STATUS.md:** Shows v0.2.2-alpha (updated 2025-01-05)
- **MLIR_PIPELINE_STATUS.md:** Shows complete pipeline (updated 2025-11-07)

**Analysis:** STATUS.md appears outdated. The project has evolved significantly:
- v0.2.2: Early implementation phase
- v0.3.1: Current state with complete MLIR pipeline

---

## Proposed Architecture: What ChatGPT Suggested

### Three-Layer Model

```
┌────────────────────────────┐
│        RiffStack           │  ← Layer 1: Performance
│  "Live patch & play"       │     YAML/RPN, stack machine,
│  Performer-facing          │     live looping
└─────────────┬──────────────┘
              │  (compiles to)
┌─────────────▼──────────────┐
│       Kairo.Audio          │  ← Layer 2: Composition
│  "Structured composition"  │     Typed DSL (Sig, Ctl, Evt)
│  Composer-facing           │     Scenes, modules
└─────────────┬──────────────┘
              │  (lowers to)
┌─────────────▼──────────────┐
│      Kairo Core            │  ← Layer 3: Kernel
│ "Deterministic kernel"     │     MLIR/LLVM, scheduler,
│  Implementor-facing        │     type system, profiles
└────────────────────────────┘
```

### Key Proposals

1. **RiffStack as Performance Layer**
   - YAML-based patch format
   - RPN (Reverse Polish Notation) stack evaluation
   - Live looping, MIDI control
   - Replace NumPy DSP with Kairo runtime

2. **Kairo.Audio as Semantic Layer**
   - Typed streams: `Sig<T>`, `Ctl<T>`, `Evt<T>`
   - Scene-based composition
   - Import/export to RiffStack YAML
   - Profile system integration

3. **Kairo Core as Kernel**
   - MLIR compilation pipeline
   - Deterministic scheduler
   - Profile system (live, render, strict modes)
   - Shared operator registry (ops.json)

4. **Shared Infrastructure**
   - `ops.json` - operator metadata registry
   - Graph JSON intermediate format
   - Hot-reload API
   - Control surface mapping

---

## Validation: Proposal vs. Reality

### ✅ What Makes Sense (Good Ideas)

#### 1. Layered Semantics Concept
**Proposal:** Different layers for different user personas (performer, composer, implementor)

**Reality:** Kairo already has this conceptually through:
- **High-level syntax** - `.kairo` files (composer-facing)
- **MLIR IR** - intermediate representation (compiler-facing)
- **Runtime** - execution engine (implementation)

**Verdict:** ✅ Good concept, already partially present

#### 2. Profile System
**Proposal:** `profile live|render|strict` for different execution modes

**Reality:** Mentioned in SPECIFICATION.md (Section 13) but not fully implemented

**Verdict:** ✅ Worth implementing, aligns with roadmap

#### 3. Shared Operator Registry
**Proposal:** Central `ops.json` for operator metadata

**Reality:** Operations are currently Python code in `stdlib/`

**Verdict:** ✅ Could improve tooling (IDE support, documentation generation)

#### 4. Hot-Reload for Live Coding
**Proposal:** Dynamic graph patching for performance

**Reality:** Mentioned as feature in README, implementation status unclear

**Verdict:** ✅ Valuable feature for creative workflow

### ❌ What Doesn't Match Reality

#### 1. Three Separate Layers
**Proposal:** RiffStack → Kairo.Audio → Kairo Core as distinct codebases

**Reality:**
- Kairo is one unified project
- No "Kairo.Audio" sublayer exists
- No RiffStack code in this repo
- Current architecture: Source → Parser → MLIR (not three layers)

**Verdict:** ❌ Doesn't describe current system

#### 2. RiffStack as Performance Frontend
**Proposal:** YAML/RPN stack-based DSL that compiles to Kairo.Audio

**Reality:**
- RiffStack is mentioned as separate project
- No compilation path exists
- No YAML patch format in Kairo
- Signal dialect exists but isn't "Kairo.Audio"

**Verdict:** ❌ Architectural relationship unclear/non-existent

#### 3. Kairo.Audio as Intermediate DSL
**Proposal:** Separate typed DSL layer between RiffStack and Core

**Reality:**
- Signal dialect is part of unified Kairo language
- No separate "Kairo.Audio" DSL
- No intermediate graph JSON format

**Verdict:** ❌ Layer doesn't exist

#### 4. Replacing NumPy with Kairo Runtime
**Proposal:** RiffStack should use Kairo runtime instead of NumPy DSP

**Reality:**
- Kairo runtime IS built on NumPy (for field operations)
- No separate "RiffStack runtime" to replace
- Field operations in `stdlib/field.py` use NumPy directly

**Verdict:** ❌ Misunderstands implementation

### ⚠️ Unclear / Context-Dependent

#### 1. RiffStack Project Relationship
**Proposal:** RiffStack compiles to Kairo

**Reality:** Relationship unclear from codebase

**Questions Needed:**
- Is RiffStack a real project or concept?
- If real, what's its current status?
- Should they integrate or stay separate?

#### 2. Multi-DSL Strategy
**Proposal:** Multiple DSLs (RiffStack, Kairo.Audio, Luma, Asterion) sharing Kairo Core

**Reality:** Only one DSL (Kairo) exists

**Questions Needed:**
- Is multi-DSL strategy desired?
- What's the vision for project ecosystem?

---

## Practical Recommendations

### Immediate Actions

#### 1. Clarify Project Scope
**Decision Needed:** Is Kairo intended to be:
- **Option A:** Single unified DSL (current state)
- **Option B:** Compilation target for multiple DSLs (proposed state)

**If Option A:** Update docs to remove RiffStack references or clarify separation
**If Option B:** Create architectural plan for multi-DSL support

#### 2. Update Outdated Documentation
- [ ] Update or archive `STATUS.md` (shows v0.2.2, should show v0.3.1)
- [ ] Clarify RiffStack relationship in README
- [ ] Document current architecture clearly

#### 3. Validate Mentioned Features
- [ ] Hot-reload: Is this implemented? (README says yes, code unclear)
- [ ] Profile system: Design exists (spec), implementation status?
- [ ] Agent dialect: Spec complete, implementation status?
- [ ] Signal dialect: Spec complete, implementation status?

### Near-Term Improvements (If Desired)

#### 1. Implement Profile System
Based on SPECIFICATION.md Section 13, add:
```python
# kairo/profiles.py
class Profile:
    """Execution profile (live, render, strict)"""
    precision: Precision  # f32 | f64
    determinism: Determinism  # bitexact | reproducible
    solver_config: SolverConfig
```

**Benefit:** Enables different optimization/precision trade-offs

#### 2. Operator Registry
Create shared metadata for tooling:
```json
// kairo/ops/registry.json
{
  "field.diffuse": {
    "params": {
      "field": "Field2D<T>",
      "rate": "f32",
      "dt": "f32"
    },
    "returns": "Field2D<T>",
    "description": "Diffuse field using heat equation"
  }
}
```

**Benefit:** IDE autocomplete, documentation generation, validation

#### 3. Better Hot-Reload
If hot-reload is desired:
- Implement state serialization
- Add graph patch API
- Enable live code updates without restart

**Benefit:** Better experience for interactive development

### Long-Term Considerations (If Multi-DSL Vision)

#### Only Pursue If User Confirms This Direction

1. **Create DSL Integration Layer**
   - Common graph IR (JSON or MLIR)
   - Compilation pipeline: DSL → IR → MLIR → Runtime
   - Shared type system

2. **Factor Out "Kairo Core"**
   ```
   kairo-core/        # Compilation, runtime, MLIR
   kairo-dsl/         # Kairo language frontend
   riffstack/         # Audio performance frontend (if real)
   ```

3. **Operator Registry as Bridge**
   - Single source of truth for operations
   - Multiple DSLs reference same ops
   - Type checking across DSLs

---

## Conclusion

### What to Tell the User

**Reality Check:**
- ✅ Kairo v0.3.1 is a complete, working DSL with MLIR pipeline
- ❌ The "three-layer architecture" doesn't currently exist
- ⚠️ RiffStack relationship is unclear (separate project? concept?)
- ⚠️ Some documentation is outdated (STATUS.md)

**Good Ideas from Proposal:**
- Profile system (already in spec, worth implementing)
- Operator registry for tooling
- Hot-reload improvements
- Layered semantics concept (refinement of current structure)

**Not Applicable:**
- RiffStack as frontend layer (doesn't exist in this repo)
- Kairo.Audio as intermediate DSL (doesn't exist)
- Three separate codebases (currently unified)
- Replacing NumPy (Kairo uses NumPy internally)

**Recommendation:**
1. First clarify the actual vision: Single DSL or multi-DSL target?
2. Update docs to match reality (especially STATUS.md)
3. If multi-DSL is desired, create proper architectural plan
4. Focus on implementing features that add value to current Kairo:
   - Profile system
   - Improved hot-reload
   - Operator metadata
   - Agent/Signal dialect completion

---

## Questions for Project Owner

1. **RiffStack Status:**
   - Does RiffStack actually exist as a project?
   - If yes, where is the repository?
   - Should it integrate with Kairo or stay separate?

2. **Architecture Vision:**
   - Is Kairo meant to be a single DSL?
   - Or a compilation target for multiple DSLs?
   - Are Luma, Asterion, TIA, Eidos real projects or concepts?

3. **Current Priorities:**
   - What features are most important to implement next?
   - Is live performance (audio) a priority?
   - Is visual simulation the main focus?

4. **Documentation:**
   - Can we update/archive STATUS.md?
   - Should RiffStack references stay in README?

---

## Appendix: Feature Implementation Status

Based on codebase analysis:

| Feature | Spec Status | Implementation | Tests |
|---------|------------|----------------|-------|
| Parser | ✅ Complete | ✅ Complete | ✅ Passing |
| Type System | ✅ Complete | ✅ Complete | ✅ Passing |
| MLIR Pipeline | ✅ Complete | ✅ Complete | ✅ 72/72 |
| Flow Blocks | ✅ Complete | ✅ Complete | ✅ Passing |
| Field Dialect | ✅ Complete | ✅ Working | ✅ Passing |
| Visual Dialect | ✅ Complete | ✅ Basic | ✅ Passing |
| Agent Dialect | ✅ In Spec | ⚠️ Partial | ⚠️ Limited |
| Signal Dialect | ✅ In Spec | ⚠️ Partial | ⚠️ Limited |
| Profile System | ✅ In Spec | ❌ Not impl | ❌ No tests |
| Hot-Reload | ⚠️ Mentioned | ❌ Unknown | ❌ No tests |
| RNG (Philox) | ✅ In Spec | ⚠️ Uses NumPy | ✅ Passing |

**Legend:**
- ✅ Complete and working
- ⚠️ Partial or unclear
- ❌ Not implemented

---

**Document Status:** Complete
**Next Action:** Review with project owner for validation and direction
