# Kairo Documentation

Welcome to the Kairo documentation! This guide will help you navigate the documentation based on what you want to accomplish.

## Quick Start

- **New to Kairo?** Start with [Getting Started](getting-started.md)
- **Need help?** Check [Troubleshooting](troubleshooting.md)

## Documentation Structure

### 📐 [Architecture](architecture/)
High-level design and architectural principles
- [Overview](architecture/overview.md) - Core Kairo architecture
- [Domain Architecture](architecture/domain-architecture.md) - How domains fit together (110KB - comprehensive!)
- [GPU & MLIR Principles](architecture/gpu-mlir-principles.md) - GPU execution and MLIR integration
- [Interactive Visualization](architecture/interactive-visualization.md) - Visualization approach

### 📋 [Specifications](specifications/)
Detailed technical specifications (19 documents)
- **Language**: [KAX Language](specifications/kax-language.md), [Type System](specifications/type-system.md)
- **Infrastructure**: [Graph IR](specifications/graph-ir.md), [MLIR Dialects](specifications/mlir-dialects.md), [Operator Registry](specifications/operator-registry.md), [Scheduler](specifications/scheduler.md), [Transform](specifications/transform.md)
- **Domains**: [Chemistry](specifications/chemistry.md), [Circuit](specifications/circuit.md), [Emergence](specifications/emergence.md), [Procedural Generation](specifications/procedural-generation.md), [Physics](specifications/physics-domains.md), [BI](specifications/bi-domain.md), [Video/Audio Encoding](specifications/video-audio-encoding.md)
- **Other**: [Geometry](specifications/geometry.md), [Coordinate Frames](specifications/coordinate-frames.md), [Profiles](specifications/profiles.md), [Snapshot ABI](specifications/snapshot-abi.md), [Timbre Extraction](specifications/timbre-extraction.md)

### 📝 [Architecture Decision Records (ADRs)](adr/)
Why key architectural decisions were made (8 records)
- [001: Unified Reference Model](adr/001-unified-reference-model.md)
- [002: Cross-Domain Architectural Patterns](adr/002-cross-domain-architectural-patterns.md)
- [003: Circuit Modeling Domain](adr/003-circuit-modeling-domain.md)
- [004: Instrument Modeling Domain](adr/004-instrument-modeling-domain.md)
- [005: Emergence Domain](adr/005-emergence-domain.md)
- [006: Chemistry Domain](adr/006-chemistry-domain.md)
- [007: GPU-First Domains](adr/007-gpu-first-domains.md)
- [008: Procedural Generation Domain](adr/008-procedural-generation-domain.md)

### 📖 [Guides](guides/)
How-to documentation for implementers
- [Domain Implementation Guide](guides/domain-implementation.md) - How to add new domains to Kairo

### 💡 [Examples](examples/)
Complete working examples demonstrating Kairo capabilities
- [Emergence Cross-Domain](examples/emergence-cross-domain.md)
- [J-Tube Firepit Multiphysics](examples/j-tube-firepit-multiphysics.md)
- [Kerbal Space Program Simulation](examples/kerbal-space-program.md)
- [Racing AI Pipeline](examples/racing-ai-pipeline.md)

### 🎯 [Use Cases](use-cases/)
Specific real-world applications
- [2-Stroke Muffler Modeling](use-cases/2-stroke-muffler-modeling.md)
- [Chemistry Unified Framework](use-cases/chemistry-unified-framework.md)

### 📚 [Reference](reference/)
Catalogs, operator references, and domain overviews
- **Operator Catalogs**: [Emergence](reference/emergence-operators.md), [Procedural](reference/procedural-operators.md), [Genetic Algorithms](reference/genetic-algorithm-operators.md), [Optimization](reference/optimization-algorithms.md), [Time Alignment](reference/time-alignment-operators.md)
- **Patterns**: [Multiphysics Success Patterns](reference/multiphysics-success-patterns.md) - 12 battle-tested architectural patterns
- **Domain Overviews**: [Procedural Graphics Domains](reference/procedural-graphics-domains.md), [Professional Domains](reference/professional-domains.md), [Visual Scene Domain](reference/visual-scene-domain.md), [Visual Domain Quick Reference](reference/visual-domain-quickref.md)
- [Operator Registry Expansion](reference/operator-registry-expansion.md)

### 🗺️ [Roadmap](roadmap/)
Planning and progress tracking
- [MVP Roadmap](roadmap/mvp.md)
- [Kairo Core v0.1 Roadmap](roadmap/v0.1.md)
- [Implementation Progress](roadmap/implementation-progress.md)
- [Testing Strategy](roadmap/testing-strategy.md)

### 📦 [Archive](archive/)
Historical documents and old reviews (well-organized for reference)

### 🏛️ [Legacy](legacy/)
Deprecated CCDSL v0.2.2 documentation (for historical reference)

---

## Finding What You Need

**I want to...**

- **Understand Kairo's architecture** → Start with [Architecture Overview](architecture/overview.md), then [Domain Architecture](architecture/domain-architecture.md)
- **Implement a new domain** → Read [Domain Implementation Guide](guides/domain-implementation.md) and relevant [ADRs](adr/)
- **Learn about a specific domain** → Check [Specifications](specifications/) for the domain spec, then related [ADRs](adr/)
- **See Kairo in action** → Browse [Examples](examples/) and [Use Cases](use-cases/)
- **Find specific operators** → Search [Reference](reference/) for operator catalogs
- **Understand a design decision** → Look in [ADRs](adr/)
- **Track project progress** → See [Roadmap](roadmap/)
- **Debug an issue** → Start with [Troubleshooting](troubleshooting.md)

---

## Recent Changes

This documentation was reorganized on 2025-11-15 to improve discoverability and navigation. Key improvements:

- ✅ Consistent lowercase naming throughout
- ✅ Logical grouping by user intent (why/what/how/how-to)
- ✅ Fixed ADR numbering (resolved duplicate 003 and 005 issues)
- ✅ All specifications in one directory for easy discovery
- ✅ Clear navigation paths for different user types
- ✅ Moved patterns catalog out of ADRs to reference section

All file history has been preserved using `git mv`.
