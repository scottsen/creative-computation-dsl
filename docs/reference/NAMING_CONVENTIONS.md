# Kairo Naming Conventions

**Quick Reference for Contributors**

This document provides quick lookup for naming patterns across Kairo. For detailed rationale, see [ADR-010](../adr/010-ecosystem-branding-naming-strategy.md).

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────┐
│  Layer 1: User Surfaces                     │
│  Kairo.Audio, RiffStack                     │
│  Pattern: Kairo.X (capitalized)             │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│  Layer 2: Domain Libraries                  │
│  field, agent, audio, rigid, linalg...      │
│  Pattern: lowercase, single word            │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│  Layer 3: Kernel (Internal)                 │
│  Stream, Field, scheduler, transform...     │
│  Pattern: kairo.internal namespace          │
└─────────────────────────────────────────────┘
```

---

## Domain Naming Rules

When adding a new domain library:

1. ✅ **Single word** - `rigid` not `rigidbody`
2. ✅ **Lowercase** - `linalg` not `LinAlg`
3. ✅ **No underscores** - `linalg` not `sparse_linalg`
4. ✅ **Noun form** - Domain objects, not implementation details
5. ✅ **Singular** - `integrator` not `integrators`
6. ✅ **Domain-first** - What domain, not how implemented

**Examples:**
```
✅ field        ❌ dense_field, grid_operations
✅ rigid        ❌ rigidbody, rigid_body
✅ linalg       ❌ sparse_linalg, linear_algebra
✅ optimize     ❌ optimization, optimizer
✅ acoustic     ❌ acoustics, acoustic_simulation
```

---

## Domain Tiers (Organizational)

**Tier 1: Core Computational Primitives**
- `field` - Dense grids
- `agent` - Sparse particles
- `stream` - Time-series
- `event` - Event sequences

**Tier 2: Physical Simulation**
- `fluid`, `thermal`, `rigid`, `acoustic`, `circuit`, `optics`

**Tier 3: Signal & Media Processing**
- `audio`, `visual`, `signal`, `image`, `video`, `color`

**Tier 4: Mathematical & Computational**
- `linalg`, `optimize`, `graph`, `neural`, `symbolic`

**Tier 5: Generative & Procedural**
- `noise`, `terrain`, `fractal`, `palette`, `cellular`

**Tier 6: System & Infrastructure**
- `io`, `state`, `integrator`

---

## Import Patterns

### For Users (Kairo Language)
```kairo
use field, agent, audio
```

### For Python (Library Usage)
```python
# Beginner
from kairo.stdlib import field, agent, audio

# Intermediate
from kairo import Stream, Field          # Core types
from kairo.stdlib import field, audio    # Domains

# Advanced
from kairo.internal import scheduler, transform
from kairo.stdlib import field, audio
```

---

## Type Names

Core types use PascalCase:
- `Stream<T, Domain, Rate>`
- `Field<T, Space>`
- `Evt<A>`
- `Agents<T>`

---

## File Naming

Domain library files match domain names:
```
kairo/stdlib/
├── field.py         # Domain: field
├── agent.py         # Domain: agent
├── audio.py         # Domain: audio
└── rigid.py         # Domain: rigid (not rigidbody.py)
```

---

## Current Renames (v0.11 → v1.0)

| Old Name | New Name | Migration Status |
|----------|----------|------------------|
| `rigidbody` | `rigid` | Alias in v0.11, rename in v1.0 |
| `sparse_linalg` | `linalg` | Alias in v0.11, rename in v1.0 |
| `statemachine` | `state` | Alias in v0.11, rename in v1.0 |
| `integrators` | `integrator` | Alias in v0.11, rename in v1.0 |
| `io_storage` | `io` | Alias in v0.11, rename in v1.0 |
| `optimization` | `optimize` | Alias in v0.11, rename in v1.0 |
| `acoustics` | `acoustic` | Alias in v0.11, rename in v1.0 |

**Both names work in v0.11+**, old names deprecated in v1.0.

---

## Quick Decision Tree

**Adding a new domain?**

1. What is the primary domain concept? (fluid, optics, symbolic, etc.)
2. Can it be a single word? If not, what's the core noun?
3. Is it lowercase?
4. Does it follow the pattern of existing domains?

**Examples:**
- Chemistry domain → `chem` or `molecule`
- Robotics domain → `robot` or `control`
- Quantum simulation → `quantum`
- Distributed computing → `distrib` or `cluster`

**When in doubt:** Prefer shorter, domain-first naming.

---

## User Surface Naming

**Pattern:** `Kairo.X` for DSL surfaces

```
Kairo.Audio      # Compositional audio DSL
Kairo.Physics    # Physics simulation DSL (future)
Kairo.Visual     # Visual composition DSL (future)
```

**Not a DSL?** Use separate brand (like RiffStack) or integrate as domain library.

---

## Anti-Patterns to Avoid

❌ **Compound words**: `rigidbody`, `statemachine`
❌ **Underscores**: `sparse_linalg`, `io_storage`
❌ **Plural forms**: `integrators`, `acoustics`
❌ **Implementation details**: `sparse_linalg` → just `linalg`
❌ **Adjectives**: `optimization` → verb form `optimize`
❌ **Long names**: `linear_algebra` → short `linalg`

---

## Questions?

See full rationale in [ADR-010: Ecosystem Branding & Naming Strategy](../adr/010-ecosystem-branding-naming-strategy.md)
