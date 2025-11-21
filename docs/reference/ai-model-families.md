# AI Model Families: Mental Models & Domain Alignment

**Version:** 1.0
**Status:** Reference Guide
**Last Updated:** 2025-11-21

---

## Overview

This document maps modern AI model families to their **deep mental models** — the types of structure they have internalized and the representations they manipulate. Understanding what each family has "truly mastered" helps us identify which AI approaches complement Morphogen's computational domains.

**Purpose:**
- Understand what different AI model families have internalized
- Map AI approaches to Morphogen's domain architecture
- Guide integration of neural methods with traditional computation
- Inform hybrid systems design (symbolic + neural)

**Prerequisites:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design foundations
- [Domain Architecture](../architecture/domain-architecture.md) — Morphogen's domain taxonomy
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Computational models

**Related:**
- [Neural Network Domain](../specifications/neural-networks.md) — Morphogen's neural operators (planned)
- [Hybrid Systems](../architecture/continuous-discrete-semantics.md#hybrid-systems) — Mixing symbolic and neural

---

## The Core Insight

**AI models are not universal reasoners — they are specialized structure learners.**

Each family has mastered a particular type of structure:
- **LLMs** → Latent grammar of symbolic thought
- **Diffusion** → Visual manifold geometry
- **World Models** → Temporal dynamics and causality
- **RL Agents** → Policies and reward optimization
- **GNNs** → Relational topology
- **NeRFs** → Volumetric radiance fields
- **NCAs** → Local rules → emergent patterns
- **AlphaFold/AlphaZero** → Optimal structure search
- **Self-Supervised Vision** → Invariant visual semantics
- **Multimodal** → Cross-domain concept alignment

**The key question:** Which structural insights map to which Morphogen domains?

---

## 1. Large Language Models (LLMs)

### Deep Mental Model

LLMs have learned the **latent grammar of thought** — not just syntax, but patterns of:
- Reasoning and argumentation
- Procedures and workflows
- World knowledge and inference
- Instructions and goals
- Code execution patterns

**They are:** Universal structure predictors in symbolic space

### What They've Truly Mastered

**✓ Latent semantic geometry**
- What concepts cluster together
- Analogical reasoning
- Conceptual interpolation

**✓ Procedural description**
- Algorithms, workflows, plans
- Step-by-step decomposition
- Execution traces

**✓ Intent expression**
- How humans express goals
- Ambiguity resolution
- Context sensitivity

**✓ Schema induction**
- Automatic format inference
- Role and constraint discovery
- Pattern abstraction

**✓ Compositional reasoning**
- Building structured wholes from parts
- Hierarchical decomposition
- Modular combination

### Morphogen Domain Alignment

| Morphogen Domain | LLM Strength | Use Case |
|-----------------|--------------|----------|
| **Procedural Generation** | High | Generating rule sets, pattern descriptions |
| **State Machines** | High | Describing transition logic, behaviors |
| **Symbolic Math** | Medium | Mathematical reasoning, proof sketches |
| **BI/Analytics** | High | Natural language → query translation |
| **Code Generation** | High | DSL generation, operator synthesis |

### Integration Opportunities

**🔹 Natural Language → Morphogen Programs**
```morphogen
// LLM translates: "Create a field that diffuses heat over time"
use field

@state temp : Field2D<f32 [K]> = random_normal(seed=42, shape=(256, 256))

flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
}
```

**🔹 Intent-Driven Domain Translation**
```morphogen
// LLM infers translation semantics from description
translate agents -> field {
    method: "kernel_density_estimation"
    preserves: {total_mass, center_of_mass}  // LLM inferred from intent
    drops: {individual_identity}
}
```

**Cross-Domain Upside:** LLMs act as a linguistic reasoning engine you can plug other models into.

---

## 2. Diffusion & Flow Models

### Deep Mental Model

Diffusion models understand the **geometry of visual manifolds** — how pixels relate across:
- Space (edges, shapes, objects, scenes)
- Style (brushwork, texture, palette, lighting)
- Material properties
- Physical constraints

**They are:** Creativity engines that understand visual structure as probability flow

### What They've Truly Mastered

**✓ Visual compositionality**
- Objects from parts
- Scenes from objects
- Styles from primitives

**✓ Style continuity**
- Brushwork physics
- Texture coherence
- Palette harmony
- Camera lens properties

**✓ Natural co-occurrence**
- What objects appear together
- Contextual plausibility
- Spatial relationships

**✓ Spatial realism**
- Lighting consistency
- Perspective correctness
- Occlusion handling

**✓ Local → Global consistency**
- Fine detail matches global structure
- Multi-scale coherence

### Morphogen Domain Alignment

| Morphogen Domain | Diffusion Strength | Use Case |
|-----------------|-------------------|----------|
| **Visual/Graphics** | Very High | Procedural texture generation |
| **Procedural Generation** | High | Content generation, variation |
| **Image Processing** | High | Style transfer, enhancement |
| **Noise Domain** | Medium | Structured noise generation |

### Integration Opportunities

**🔹 Procedural Texture Synthesis**
```morphogen
use procedural, image, diffusion

// Generate texture from description
let texture = diffusion.text_to_texture(
    prompt="rusty metal with scratches",
    size=(1024, 1024),
    seed=42
)

// Apply to geometry
let material = procedural.material(
    albedo=texture,
    roughness=0.8,
    metallic=1.0
)
```

**🔹 Field Visualization Enhancement**
```morphogen
use field, visual, diffusion

@state temp : Field2D<f32> = simulate_heat()

flow() {
    // Traditional visualization
    let viz_basic = visual.colorize(temp, palette="fire")

    // Diffusion-enhanced (stylized)
    let viz_enhanced = diffusion.enhance_visualization(
        viz_basic,
        style="scientific illustration"
    )
}
```

**Cross-Domain Upside:** Diffusion models bring creative visual structure to deterministic simulation.

---

## 3. World Models (Latent Dynamics)

### Deep Mental Model

World models internalize **dynamics, not snapshots** — the rules that govern change:
- Physics constraints
- Causality
- Temporal dependencies
- Agent/environment interactions

**They are:** Compressed simulators that can roll out futures without touching the real world

### What They've Truly Mastered

**✓ Latent video prediction**
- Implied next states
- Trajectory forecasting

**✓ Transition structures**
- How situations become other situations
- State space topology

**✓ Cause-effect chains**
- Causal inference
- Intervention effects

**✓ Counterfactual reasoning**
- "What if X had occurred?"
- Alternative timelines

**✓ Model-based foresight**
- Planning in latent space
- Efficient search

### Morphogen Domain Alignment

| Morphogen Domain | World Model Strength | Use Case |
|-----------------|---------------------|----------|
| **Physics (RigidBody)** | High | Learned dynamics approximation |
| **Agents** | High | Behavior prediction |
| **Field** | Medium | Fast surrogate models |
| **Control Systems** | High | Model predictive control |

### Integration Opportunities

**🔹 Fast Surrogate Physics**
```morphogen
use physics, world_model

@state bodies : RigidBodyWorld = initialize()
@state learned_model : WorldModel = train_from_physics(bodies, episodes=1000)

flow(dt=0.016) {
    // Fast prediction using learned dynamics
    let predicted_states = learned_model.predict(bodies, horizon=10)

    // Occasionally verify with true physics
    if step % 100 == 0:
        bodies = physics.step(bodies)  // True dynamics
}
```

**🔹 Agent Behavior Prediction**
```morphogen
use agent, world_model

@state agents : Agents<Boid> = alloc(count=100)
@state model : AgentWorldModel = learn_dynamics(agents)

flow(dt=0.01) {
    // Predict future agent states
    let future_positions = model.predict(agents, steps=50)

    // Use for planning/optimization
    let optimal_action = optimize_based_on_future(future_positions)
}
```

**Cross-Domain Upside:** World models are "little universes" that can run inside an agent's head.

---

## 4. Reinforcement Learning Agents

### Deep Mental Model

RL systems internalize **policies, not predictions** — they learn how to act, not just what comes next.

Their representation is a **skill manifold** encoding:
- Strategies
- Habits
- Long-horizon dependencies
- Reward gradients
- Trade-offs

**They are:** Adaptive behavior machines that optimize within environments

### What They've Truly Mastered

**✓ Temporal credit assignment**
- Delayed rewards
- Multi-step consequences

**✓ Exploration vs. exploitation**
- Information gathering
- Optimal action selection

**✓ Policy compression**
- Strategy generalization
- Skill abstraction

**✓ Feedback loop adaptation**
- Online learning
- Environmental changes

**✓ Emergent skill chaining**
- Primitive → complex behaviors
- Hierarchical policies

### Morphogen Domain Alignment

| Morphogen Domain | RL Strength | Use Case |
|-----------------|------------|----------|
| **Optimization** | Very High | Parameter tuning, design optimization |
| **Agents** | Very High | Autonomous behavior learning |
| **Game AI** | Very High | Strategy learning |
| **Control Systems** | High | Controller synthesis |

### Integration Opportunities

**🔹 Learned Control Policies**
```morphogen
use physics, control, rl

@state pendulum : InvertedPendulum = create()
@state controller : RLPolicy = train_controller(
    environment=pendulum,
    reward=keep_upright,
    episodes=10000
)

flow(dt=0.01) {
    // RL-learned control
    let action = controller.act(pendulum.state)
    pendulum = pendulum.apply_force(action)
}
```

**🔹 Design Optimization**
```morphogen
use geometry, optimization, rl

@state design : ParametricDesign = initial_design()
@state optimizer : RLOptimizer = create_optimizer(
    objective=minimize_drag,
    constraints=[strength > 1000Pa, mass < 5kg]
)

// RL explores design space
for episode in 1..1000:
    design = optimizer.suggest_design()
    performance = simulate(design)
    optimizer.update(performance)
```

**Cross-Domain Upside:** RL agents are ideal for any domain where improvement comes from interaction.

---

## 5. Graph Neural Networks (GNNs)

### Deep Mental Model

GNNs understand **structure as relationships** — they operate on:
- Nodes
- Edges
- Topology
- Message passing

**They are:** Relational reasoning machines that compute on structured data

### What They've Truly Mastered

**✓ Interaction patterns**
- Chemical bonds
- Social networks
- Citation graphs
- Molecular structure

**✓ Equivariance to structure**
- Permutation invariance
- Structural symmetries

**✓ Causal graph inference**
- Discovering dependencies
- Causal relationships

**✓ Distributed computation**
- Message passing
- Belief propagation

**✓ Combinatorial generalization**
- Novel graph structures
- Zero-shot transfer

### Morphogen Domain Alignment

| Morphogen Domain | GNN Strength | Use Case |
|-----------------|-------------|----------|
| **Graph/Network** | Very High | Network analysis, property prediction |
| **Chemistry** | Very High | Molecular property prediction |
| **Social Networks** | High | Influence prediction, community detection |
| **Circuit Design** | Medium | Circuit optimization |

### Integration Opportunities

**🔹 Molecular Property Prediction**
```morphogen
use chemistry, graph, gnn

@state molecule : Molecule = create_from_smiles("CCO")
@state graph : MolecularGraph = molecule_to_graph(molecule)

// GNN predicts properties
let properties = gnn.predict_properties(graph)
// {solubility: 0.8, toxicity: 0.1, drug_likeness: 0.7}
```

**🔹 Network Optimization**
```morphogen
use graph, gnn

@state network : Graph = create_network(nodes=1000, edges=5000)

// GNN suggests edge additions/removals for optimization
let suggestions = gnn.optimize_topology(
    network,
    objective=minimize_diameter,
    constraints=[degree < 10]
)
```

**Cross-Domain Upside:** GNNs are the best models for problems where the connection pattern is the data.

---

## 6. NeRFs & Implicit Neural Representations

### Deep Mental Model

NeRFs understand the **continuous structure of 3D space** — they encode scenes as volumetric functions, not meshes.

They unify:
- Geometry
- Appearance
- Viewpoint
- Lighting

**They are:** Scene priors embedded in neural fields

### What They've Truly Mastered

**✓ Multi-view consistency**
- Coherent 3D from 2D views
- View synthesis

**✓ Implicit geometry inference**
- Surfaces from samples
- Density fields

**✓ Photometric continuity**
- Realistic appearance
- View-dependent effects

**✓ Scene interpolation**
- What lives between views
- Smooth trajectories

**✓ Compact 3D encoding**
- Efficient scene representation
- Continuous queries

### Morphogen Domain Alignment

| Morphogen Domain | NeRF Strength | Use Case |
|-----------------|--------------|----------|
| **Geometry** | High | Scene reconstruction, novel view synthesis |
| **Graphics** | High | Real-time rendering, avatars |
| **Physics** | Medium | Implicit collision detection |
| **Robotics** | High | Scene understanding, navigation |

### Integration Opportunities

**🔹 Geometry Reconstruction**
```morphogen
use geometry, nerf

// Reconstruct 3D scene from images
let images = load_image_sequence("scene/*.jpg")
let scene = nerf.reconstruct_scene(images, epochs=1000)

// Query implicit geometry
let surface = scene.extract_mesh(threshold=0.5)
let material = scene.extract_appearance()
```

**🔹 Real-Time Rendering**
```morphogen
use visual, nerf

@state scene : NeRFScene = load_trained_nerf("scene.pth")

flow(dt=0.016) {  // 60 FPS
    let camera = get_camera_pose()
    let rendered = scene.render(camera, quality="fast")
    visual.display(rendered)
}
```

**Cross-Domain Upside:** NeRFs enable fast reconstruction, navigation, and simulation of 3D scenes.

---

## 7. Neural Cellular Automata (NCA)

### Deep Mental Model

NCAs understand **local rules → global behavior** — they're experts in emergence.

They treat structure as:
- Cells
- States
- Local interactions
- Self-organization

**They are:** Neural organisms that grow and self-repair

### What They've Truly Mastered

**✓ Morphogenesis-like growth**
- Pattern development
- Form emergence

**✓ Self-repair**
- Damage recovery
- Robustness

**✓ Stable attractors**
- Convergent patterns
- Homeostasis

**✓ Procedural complexity**
- Simple rules → complex outcomes

**✓ Pattern evolution**
- Temporal unfolding

### Morphogen Domain Alignment

| Morphogen Domain | NCA Strength | Use Case |
|-----------------|-------------|----------|
| **Cellular Automata** | Very High | Learned CA rules |
| **Procedural Generation** | High | Texture synthesis, pattern generation |
| **Biology/Growth** | High | Morphogenesis simulation |
| **Emergence** | High | Self-organizing systems |

### Integration Opportunities

**🔹 Learned Texture Growth**
```morphogen
use procedural, nca

// Train NCA to grow target texture
let nca = nca.train_texture_growth(
    target=load_image("target.jpg"),
    epochs=10000
)

// Grow from seed
@state texture : Field2D<RGB> = single_pixel_seed()

flow(dt=0.1, steps=100) {
    texture = nca.step(texture)
    output texture
}
```

**🔹 Self-Repairing Patterns**
```morphogen
use nca, field

@state pattern : Field2D<f32> = nca.grown_pattern()

flow(dt=0.1) {
    // Damage pattern
    if random() < 0.01:
        pattern = damage_region(pattern, size=10)

    // NCA self-repairs
    pattern = nca.step(pattern)
}
```

**Cross-Domain Upside:** NCAs behave like neural organisms, ideal for systems that grow or maintain themselves.

---

## 8. Algorithm-Discovering Models (AlphaFold, AlphaZero, etc.)

### Deep Mental Model

These models are **structure solvers** that internalize:
- Search strategies
- Constraints
- Rules and symmetries
- Optimization landscapes

**They are:** AI scientists specializing in domains with rigid underlying structure

### What They've Truly Mastered

**✓ Latent theorem spaces**
- Mathematical structures
- Proof strategies

**✓ Latent search strategies**
- Monte Carlo Tree Search
- Beam search variations
- Pruning heuristics

**✓ Games as rule-based universes**
- Legal move generation
- Position evaluation
- Strategic planning

**✓ Physical & biological constraints**
- Protein folding rules
- Chemical validity
- Physical plausibility

**✓ Algorithmic compression**
- Finding shorter solutions
- Optimal procedures

### Morphogen Domain Alignment

| Morphogen Domain | Alpha-Type Strength | Use Case |
|-----------------|-------------------|----------|
| **Optimization** | Very High | Combinatorial optimization, design |
| **Chemistry** | Very High | Protein folding, molecule design |
| **Game AI** | Very High | Strategy discovery |
| **Mathematics** | High | Theorem proving, algorithm discovery |

### Integration Opportunities

**🔹 Molecular Design**
```morphogen
use chemistry, alphafold

// Design protein to bind target
let target = load_protein("target.pdb")
let designed = alphafold.design_binder(
    target,
    constraints={stability > 0.8, affinity > 1e-9}
)

// Verify with simulation
let complex = chemistry.dock(designed, target)
assert(complex.binding_energy < threshold)
```

**🔹 Combinatorial Optimization**
```morphogen
use optimization, alphazero

// Optimize circuit layout
let problem = CircuitLayoutProblem(components=100, connections=500)
let solver = alphazero.train_solver(problem, episodes=100000)

// Find near-optimal solution
let layout = solver.solve(problem, time_limit=60s)
```

**Cross-Domain Upside:** They invent procedures that outperform human intuition in structured domains.

---

## 9. Self-Supervised Vision Models

### Deep Mental Model

These models understand **visual semantics without teaching signals** — they learn:
- Objectness (what counts as "a thing")
- Invariance (what changes ↔ what stays consistent)
- Global gestalt
- Parts/whole structure

**They are:** General-purpose visual understanding engines

### What They've Truly Mastered

**✓ Contextual prediction**
- Masked patch prediction
- Reconstruction

**✓ Feature disentangling**
- Separating factors of variation
- Attribute decomposition

**✓ Stable conceptual clusters**
- Semantic grouping
- Category discovery

**✓ Perceptual invariance**
- Lighting, pose, cropping
- View-independent features

**✓ Latent semantic hierarchies**
- Edges → textures → parts → objects → scenes

### Morphogen Domain Alignment

| Morphogen Domain | Self-Supervised Vision Strength | Use Case |
|-----------------|-------------------------------|----------|
| **Computer Vision** | Very High | Object detection, segmentation |
| **Image Processing** | High | Feature extraction, analysis |
| **Robotics** | High | Visual scene understanding |
| **Medical Imaging** | High | Anomaly detection, diagnosis |

### Integration Opportunities

**🔹 Visual Feature Extraction**
```morphogen
use vision, self_supervised

let model = load_pretrained("DINO_vitb16")

// Extract features from images
let images = load_batch("dataset/*.jpg")
let features = model.extract_features(images)

// Use for downstream tasks
let clusters = kmeans(features, k=10)
```

**🔹 Anomaly Detection**
```morphogen
use vision, self_supervised

@state model : SelfSupervisedVision = train_on_normal_data()

flow() {
    let new_image = camera.capture()
    let reconstruction = model.reconstruct(new_image)
    let anomaly_score = mse(new_image, reconstruction)

    if anomaly_score > threshold:
        alert("Anomaly detected")
}
```

**Cross-Domain Upside:** Provides general-purpose visual understanding with minimal labels.

---

## 10. Multimodal Foundation Models

### Deep Mental Model

These models internalize **cross-domain alignment** — they learn a shared latent space where:

```
Text ↔ Image ↔ Audio ↔ Video ↔ Actions
```

all map into interpretable relationships.

**They are:** Concept space unifiers, not just "image captioners"

### What They've Truly Mastered

**✓ Cross-modal correspondence**
- Text describes images
- Images illustrate text
- Audio matches visuals

**✓ Grounding**
- Binding language to perception
- Spatial reasoning from text
- Action understanding

**✓ Instruction-following across modalities**
- "Draw this description"
- "Describe this scene"
- "Find the sound of X"

**✓ Diagram & spatial reasoning**
- Charts, graphs, diagrams
- 3D spatial relationships
- Technical drawings

**✓ Rich embeddings**
- Format-independent concepts
- Semantic similarity across modalities

### Morphogen Domain Alignment

| Morphogen Domain | Multimodal Strength | Use Case |
|-----------------|-------------------|----------|
| **All Domains** | High | Natural language interfaces |
| **Visual** | Very High | Image understanding, generation |
| **Audio** | High | Sound generation, analysis |
| **Robotics** | Very High | Embodied AI, grounding |

### Integration Opportunities

**🔹 Natural Language → Multi-Domain Programs**
```morphogen
// Multimodal model translates intent to Morphogen program
let intent = "Simulate heat spreading through a metal plate and show it as a video"

let program = multimodal.translate_to_morphogen(intent)
// Generates:
// use field, visual
// @state temp : Field2D<f32> = ...
// flow(dt=0.01) {
//     temp = diffuse(temp, rate=0.1, dt)
//     output visual.colorize(temp)
// }
```

**🔹 Cross-Modal Synthesis**
```morphogen
use audio, visual, multimodal

// Generate visuals from audio
let audio = load_audio("music.wav")
let visuals = multimodal.audio_to_visual(
    audio,
    style="abstract geometric patterns"
)

// Or vice versa
let image = load_image("scene.jpg")
let ambient_sound = multimodal.image_to_audio(
    image,
    style="ambient soundscape"
)
```

**Cross-Domain Upside:** Underpins universal assistants, robotics grounding, and complex reasoning over real-world data.

---

## Summary Table: Mental Models & Structural Insights

| Model Family | Core Structural Insight | What They've Internalized | Best Morphogen Domains |
|--------------|------------------------|--------------------------|----------------------|
| **LLMs** | Latent grammar of thought | Symbolic reasoning patterns, procedures, intent | Procedural, State Machines, BI, Code Gen |
| **Diffusion** | Visual manifold geometry | Image structure, style, composition | Visual, Procedural, Image |
| **World Models** | Temporal dynamics | Physics, causality, transitions | Physics, Agents, Control |
| **RL Agents** | Policy optimization | Strategies, skills, reward gradients | Optimization, Agents, Game AI, Control |
| **GNNs** | Relational topology | Graphs, interactions, message passing | Graph, Chemistry, Networks |
| **NeRFs** | Volumetric radiance | 3D geometry, multi-view consistency | Geometry, Graphics, Robotics |
| **NCAs** | Local → global emergence | Self-organization, morphogenesis | Cellular Automata, Procedural, Emergence |
| **AlphaFold/Zero** | Optimal structure search | Constraints, rules, search strategies | Optimization, Chemistry, Math |
| **Self-Supervised Vision** | Invariant visual semantics | Objectness, features, hierarchies | Vision, Image Processing, Robotics |
| **Multimodal** | Cross-domain alignment | Unified concept space across modalities | All domains (universal interface) |

---

## Design Principles for Neural-Symbolic Integration

### Principle 1: Use Neural Where Structure Is Implicit

**Neural wins when:**
- Structure is complex and implicit (visual, natural language)
- Hard to formalize (aesthetics, style)
- Data is abundant

**Symbolic wins when:**
- Structure is explicit and formalizable (physics, logic)
- Determinism required
- Interpretability matters

**Hybrid approach:**
```morphogen
// Neural for complex perception
let features = vision_model.extract_features(image)

// Symbolic for reasoning
let constraints = [mass_conserved, energy_conserved]
let solution = symbolic_solver.solve(features, constraints)
```

### Principle 2: Let Neural Learn What Symbolic Can't Formalize

**Example: Learned physics approximations**
```morphogen
use physics, world_model

// Symbolic physics (slow but exact)
let exact = physics.simulate(state, dt, method="rk4")

// Neural approximation (fast but approximate)
let approx = world_model.predict(state, dt)

// Use neural for exploration, verify with symbolic
if exploration_phase:
    next_state = approx
else:
    next_state = exact
```

### Principle 3: Use Symbolic to Constrain Neural

**Example: Physics-informed neural networks**
```morphogen
use field, neural

// Neural network respects physical laws
let solution = neural_field.solve_pde(
    pde=heat_equation,
    boundary_conditions=bc,
    constraints=[energy_conserved, entropy_increasing]
)

// Symbolic verification
assert(verify_conservation_laws(solution))
```

### Principle 4: Cross-Modal Grounding

**Example: Language-grounded simulation**
```morphogen
use multimodal, physics

let description = "A ball rolls down a ramp and hits a wall"

// Multimodal → scene setup
let scene = multimodal.text_to_scene(description)

// Symbolic physics simulation
let trajectory = physics.simulate(scene, duration=5s)

// Multimodal → language summary
let summary = multimodal.scene_to_text(trajectory)
// "The ball accelerated down the ramp, reaching 5 m/s before colliding with the wall"
```

---

## Future Directions

### Planned Integrations

**v0.12: Neural Operator Support**
- `use neural` domain
- Learned operators (physics surrogates, texture synthesis)
- Training integration with PyTorch/JAX

**v0.13: Multimodal Grounding**
- Natural language → Morphogen program translation
- Visual → audio synthesis
- Cross-modal property prediction

**v0.14: Hybrid Solvers**
- Physics-informed neural networks
- Neural-symbolic optimization
- Learned + analytic dynamics

**v0.15: Agent Learning**
- RL policy integration
- Learned behaviors for agent domain
- Self-play and emergent strategies

---

## Further Reading

**Morphogen Documentation:**
- [Universal DSL Principles](../philosophy/universal-dsl-principles.md) — Design foundations
- [Domain Architecture](../architecture/domain-architecture.md) — Domain taxonomy
- [Continuous-Discrete Semantics](../architecture/continuous-discrete-semantics.md) — Computational models

**External Resources:**
- **"Attention Is All You Need"** — Transformer architecture (LLMs)
- **"Denoising Diffusion Probabilistic Models"** — Diffusion foundations
- **"World Models"** — Ha & Schmidhuber
- **"Graph Neural Networks"** — Survey paper
- **"NeRF: Representing Scenes as Neural Radiance Fields"** — Mildenhall et al.
- **"Growing Neural Cellular Automata"** — Mordvintsev et al.
- **"AlphaFold: Highly accurate protein structure prediction"** — DeepMind
- **"CLIP: Learning Transferable Visual Models"** — OpenAI

---

## Summary

**AI model families are specialized structure learners:**

1. **LLMs** → Universal linguistic reasoning
2. **Diffusion** → Visual manifold geometry
3. **World Models** → Temporal dynamics
4. **RL** → Adaptive policies
5. **GNNs** → Relational reasoning
6. **NeRFs** → 3D volumetric structure
7. **NCAs** → Emergent self-organization
8. **AlphaFold/Zero** → Optimal structure search
9. **Self-Supervised Vision** → Invariant features
10. **Multimodal** → Cross-domain alignment

**The opportunity:** Integrate neural structure learning with Morphogen's symbolic computation to create **hybrid systems** that leverage the strengths of both paradigms.

---

**Next:** See [Neural Network Domain](../specifications/neural-networks.md) for planned integration details, or [Hybrid Systems](../architecture/continuous-discrete-semantics.md#hybrid-systems) for mixing symbolic and neural computation.
