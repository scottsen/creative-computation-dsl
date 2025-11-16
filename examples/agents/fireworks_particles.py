"""Fireworks particle effects demonstration.

Showcases particle emission, lifetime management, trails, and visual effects.
"""

import numpy as np
from kairo.stdlib.agents import agents, particle_behaviors
from kairo.stdlib.visual import visual


def create_firework_burst(center, color_base, count=200):
    """Create a single firework burst."""
    # Emit particles in sphere pattern
    particles = agents.emit(
        count=count,
        position=center,
        emission_shape="sphere",
        emission_radius=50.0,
        velocity=lambda n: np.random.randn(n, 2) * 3.0,  # Random velocities
        lifetime=(30.0, 60.0),  # Random lifetimes
        properties={
            'color_offset': np.random.rand(count) * 0.3,  # Color variation
            'size': np.random.uniform(1.0, 3.0, count)
        },
        seed=None  # Random seed for variety
    )

    return particles


def fireworks_demo():
    """Run fireworks particle effects demo."""

    print("Fireworks Particle Effects Demo")
    print("=" * 50)

    # Simulation parameters
    width, height = 512, 512
    dt = 1.0

    # Create initial empty particle system
    all_particles = agents.alloc(count=0, properties={
        'pos': np.empty((0, 2)),
        'vel': np.empty((0, 2)),
        'age': np.empty(0),
        'lifetime': np.empty(0)
    })

    # Firework launch timing
    firework_timer = 0
    firework_interval = 20

    # Colors for different fireworks
    firework_colors = [
        (1.0, 0.2, 0.2),  # Red
        (0.2, 1.0, 0.2),  # Green
        (0.2, 0.2, 1.0),  # Blue
        (1.0, 1.0, 0.2),  # Yellow
        (1.0, 0.2, 1.0),  # Magenta
        (0.2, 1.0, 1.0),  # Cyan
    ]

    def generate_frame():
        nonlocal all_particles, firework_timer

        frame_count = 0
        max_frames = 300

        while frame_count < max_frames:
            # Launch new fireworks periodically
            if firework_timer <= 0:
                # Random position in upper half
                launch_pos = np.array([
                    np.random.uniform(100, width - 100),
                    np.random.uniform(height * 0.3, height * 0.7)
                ])

                # Random color
                color = firework_colors[np.random.randint(len(firework_colors))]

                # Create burst
                new_burst = create_firework_burst(launch_pos, color, count=150)

                # Merge with existing particles
                if all_particles.alive_count > 0:
                    all_particles = agents.merge([all_particles, new_burst])
                else:
                    all_particles = new_burst

                firework_timer = firework_interval + np.random.randint(-5, 5)

            firework_timer -= 1

            # Apply gravity
            if all_particles.alive_count > 0:
                all_particles = agents.apply_force(
                    all_particles,
                    force=np.array([0.0, -2.0]),  # Gravity
                    dt=dt
                )

                # Apply air resistance
                all_particles = agents.apply_force(
                    all_particles,
                    force=particle_behaviors.drag(coefficient=0.02),
                    dt=dt
                )

                # Update positions
                all_particles = agents.integrate(all_particles, dt=dt)

                # Age particles and remove dead ones
                all_particles = agents.age_particles(all_particles, dt=dt)

                # Update trails
                if frame_count % 2 == 0:  # Update trails every other frame
                    all_particles = agents.update_trail(all_particles, trail_length=15)

            # Render
            if all_particles.alive_count > 0:
                # Calculate alpha based on age
                alphas = agents.get_particle_alpha(
                    all_particles,
                    fade_in=0.1,
                    fade_out=0.3
                )

                # Create temporary alpha property for rendering
                all_particles.properties['alpha'] = np.zeros(all_particles.count, dtype=np.float32)
                all_particles.properties['alpha'][all_particles.alive_mask] = alphas

                # Render with trails and additive blending
                vis = visual.agents(
                    all_particles,
                    width=width,
                    height=height,
                    alpha_property='alpha',
                    size_property='size',
                    size=2.0,
                    background=(0.0, 0.0, 0.05),  # Dark blue background
                    blend_mode='additive',
                    trail=True,
                    trail_length=15,
                    trail_alpha=0.4
                )
            else:
                # Empty frame
                vis = visual.layer(width=width, height=height, background=(0.0, 0.0, 0.05))

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Frame {frame_count}, Particles alive: {all_particles.alive_count}")

            yield vis

    # Create generator
    gen = generate_frame()

    # Export to video
    print("\nExporting fireworks animation...")
    visual.video(
        lambda: next(gen),
        path="/tmp/fireworks_particles.mp4",
        fps=30,
        max_frames=300
    )

    print(f"\nFireworks animation saved to /tmp/fireworks_particles.mp4")
    print(f"Total frames: 300")


if __name__ == "__main__":
    fireworks_demo()
