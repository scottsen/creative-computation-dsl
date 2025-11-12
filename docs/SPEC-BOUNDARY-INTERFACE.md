# SPEC: Boundary & Interface

## Boundary

Boundaries define how fields behave at domain edges and are **first-class typed objects** in Kairo.

### Syntax

```kairo
boundary bc = Boundary {
  left: inflow(expr),
  right: outflow,
  top: reflect,
  bottom: noslip
}
```

### Boundary Conditions

#### Standard Conditions

- `dirichlet(value)` — Fixed value at boundary
- `neumann(value)` — Fixed derivative at boundary
- `periodic` — Periodic boundary (wraps to opposite side)
- `reflect` — Reflective boundary (mirrors values)
- `noslip` — No-slip condition (velocity = 0 at boundary)
- `freeslip` — Free-slip condition (tangential velocity preserved)
- `outflow` — Open boundary (zero gradient)
- `inflow(expr)` — Prescribed inflow condition

#### Time-Varying Boundaries

Boundaries can be driven by signals:

```kairo
# Pulsating inflow
let pulse = sine(freq=2Hz) * 0.5 + 0.5
boundary bc = Boundary {
  left: inflow(pulse),
  right: outflow
}
```

### Attachment to Streams/Spaces

Boundaries are attached to streams or spaces:

```kairo
@state vel: Field2D<Vec2<f32>> = zeros((256, 256), boundary=bc)

# Or explicitly:
vel = vel.with_boundary(bc)
```

### Solver Contracts

Solvers declare how they handle boundaries:

```kairo
BoundaryContract {
  apply: pre | post,      // When to apply boundary conditions
  mode: ghost | mirror    // How to implement (ghost cells vs. mirroring)
}
```

**Example:**

```kairo
# Pressure solver declares contract
solver.pressure.contract = BoundaryContract {
  apply: pre,
  mode: ghost
}
```

The compiler/scheduler ensures boundary conditions are applied at the right time.

## Interface

Interfaces couple fields across **different domains or solvers**, enabling multi-physics simulations.

### Syntax

```kairo
interface coupling = Interface {
  space_a: fluid_domain,
  space_b: thermal_domain,
  rule: flux_match
}
```

### Interface Rules

#### Standard Rules

- `continuous` — Values continuous across interface (C0)
- `flux_match` — Fluxes match across interface (e.g., heat flux, mass flux)
- `insulated` — Zero flux across interface
- `custom(fn)` — User-defined coupling function

#### Example: Fluid-Thermal Coupling

```kairo
# Fluid domain
@state vel: Field2D<Vec2<f32>> = zeros((256, 256))
@state temp_fluid: Field2D<f32> = uniform(300K)

# Thermal domain (different resolution/grid)
@state temp_solid: Field2D<f32> = uniform(350K)

# Interface: continuous temperature, matched heat flux
interface fluid_solid = Interface {
  space_a: fluid_domain,
  space_b: solid_domain,
  rule: flux_match,
  variable: temp
}

flow sim(dt=0.01) {
  # Solve fluid with convection
  vel = navier_stokes(vel, temp_fluid, dt)
  temp_fluid = advect(temp_fluid, vel, dt)

  # Solve solid heat conduction
  temp_solid = diffuse(temp_solid, rate=KAPPA_SOLID, dt)

  # Enforce interface coupling
  (temp_fluid, temp_solid) = apply_interface(fluid_solid, temp_fluid, temp_solid)

  output colorize(temp_fluid)
}
```

### Grid Alignment

Interfaces handle **different grids and resolutions**:

```kairo
interface coupling = Interface {
  space_a: fine_grid,    // 512×512
  space_b: coarse_grid,  // 128×128
  rule: continuous,
  interpolation: linear   // How to transfer between grids
}
```

## Propagation into Solvers

Both boundaries and interfaces are **propagated** into solver calls:

```kairo
# Boundary automatically used by solver
vel = project(vel, boundary=bc, method="cg")

# Interface automatically used by multi-domain solver
(temp_a, temp_b) = coupled_diffusion(
  temp_a, temp_b,
  interface=coupling,
  dt
)
```

## Deterministic Application

Boundary and interface applications are **deterministic** and **reproducible**:

- Ghost cell values computed using deterministic interpolation.
- Interface transfers use deterministic resampling (same modes as multirate scheduler).
- Order of application fixed by `BoundaryContract`.

## Visualization

Boundaries and interfaces can be visualized for debugging:

```kairo
output visualize(bc, field=vel)
output visualize(coupling, fields=(temp_a, temp_b))
```

## Examples

### 1. Reflecting Box

```kairo
boundary box = Boundary {
  left: reflect,
  right: reflect,
  top: reflect,
  bottom: reflect
}

@state particles: Agents<Particle> = spawn(1000)

flow sim(dt=0.01) {
  particles = integrate(particles, dt)
  particles = apply_boundary(particles, box)  // Reflect at edges
}
```

### 2. Periodic Domain

```kairo
boundary periodic_bc = Boundary {
  left: periodic,
  right: periodic,
  top: periodic,
  bottom: periodic
}

@state density: Field2D<f32> = random(seed=42, shape=(256, 256), boundary=periodic_bc)

flow sim(dt=0.01) {
  density = advect(density, vel, dt)  // Wraps at edges
}
```

### 3. Multi-Domain PDE

```kairo
# Air domain
@state temp_air: Field2D<f32> = uniform(293K)

# Ground domain (1D)
@state temp_ground: Field1D<f32> = uniform(283K)

# Interface: ground-air heat exchange
interface ground_air = Interface {
  space_a: ground_space,
  space_b: air_space,
  rule: flux_match,
  orientation: horizontal
}

flow coupled_sim(dt=0.1) {
  temp_air = diffuse(temp_air, rate=KAPPA_AIR, dt)
  temp_ground = diffuse(temp_ground, rate=KAPPA_GROUND, dt)

  # Couple via interface
  (temp_ground, temp_air) = apply_interface(ground_air, temp_ground, temp_air)
}
```

## Open Questions

1. **Curved Boundaries**: How to represent non-axis-aligned boundaries?
2. **Moving Boundaries**: Time-varying geometry (e.g., moving piston)?
3. **Contact Interfaces**: Handling contact/separation in multi-body dynamics?
