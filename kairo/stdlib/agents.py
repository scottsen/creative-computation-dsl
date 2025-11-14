"""Agent operations implementation using NumPy backend.

This module provides NumPy-based implementations of all core agent operations
for sparse particle/agent-based modeling, including allocation, mapping, filtering,
force calculations, and field-agent coupling.
"""

from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np


class Agents:
    """Sparse agent collection with per-agent properties.

    Represents a collection of agents where each agent has multiple properties
    (position, velocity, mass, etc.) stored as separate NumPy arrays for efficient
    vectorized operations.

    Example:
        agents = Agents(
            count=1000,
            properties={
                'pos': np.random.rand(1000, 2),  # 2D positions
                'vel': np.zeros((1000, 2)),       # 2D velocities
                'mass': np.ones(1000)             # Scalar masses
            }
        )
    """

    def __init__(self, count: int, properties: Dict[str, np.ndarray]):
        """Initialize agent collection.

        Args:
            count: Number of agents
            properties: Dictionary mapping property names to NumPy arrays
                       Each array's first dimension must equal count
        """
        self.count = count
        self.properties = properties
        self.alive_mask = np.ones(count, dtype=bool)

        # Validate properties
        for name, arr in properties.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Property '{name}' must be a NumPy array")
            if arr.shape[0] != count:
                raise ValueError(
                    f"Property '{name}' has {arr.shape[0]} elements, expected {count}"
                )

    @property
    def alive_count(self) -> int:
        """Get number of currently alive agents."""
        return np.sum(self.alive_mask)

    def get(self, property_name: str) -> np.ndarray:
        """Get property array for all alive agents.

        Args:
            property_name: Name of property to retrieve

        Returns:
            NumPy array of property values for alive agents only

        Raises:
            KeyError: If property doesn't exist
        """
        if property_name not in self.properties:
            raise KeyError(f"Unknown property: {property_name}")
        return self.properties[property_name][self.alive_mask]

    def get_all(self, property_name: str) -> np.ndarray:
        """Get property array for ALL agents (including dead ones).

        Args:
            property_name: Name of property to retrieve

        Returns:
            NumPy array of property values for all agents
        """
        if property_name not in self.properties:
            raise KeyError(f"Unknown property: {property_name}")
        return self.properties[property_name]

    def set(self, property_name: str, values: np.ndarray) -> 'Agents':
        """Set property array for alive agents.

        Args:
            property_name: Name of property to set
            values: New values (length must match alive_count)

        Returns:
            self (for chaining)

        Raises:
            KeyError: If property doesn't exist
            ValueError: If values shape doesn't match alive agents
        """
        if property_name not in self.properties:
            raise KeyError(f"Unknown property: {property_name}")

        alive_indices = np.where(self.alive_mask)[0]
        if len(values) != len(alive_indices):
            raise ValueError(
                f"Expected {len(alive_indices)} values, got {len(values)}"
            )

        self.properties[property_name][self.alive_mask] = values
        return self

    def update(self, property_name: str, values: np.ndarray) -> 'Agents':
        """Update property for alive agents (alias for set, matches Kairo syntax).

        Args:
            property_name: Name of property to update
            values: New values

        Returns:
            New Agents instance with updated property
        """
        # Create a copy to maintain immutability (like Field2D.copy())
        new_agents = self.copy()
        new_agents.set(property_name, values)
        return new_agents

    def copy(self) -> 'Agents':
        """Create a deep copy of this agent collection.

        Returns:
            New Agents instance with copied data
        """
        return Agents(
            count=self.count,
            properties={name: arr.copy() for name, arr in self.properties.items()}
        )

    def __repr__(self) -> str:
        """String representation of agents."""
        props = ', '.join(self.properties.keys())
        return f"Agents(count={self.alive_count}/{self.count}, properties=[{props}])"


class AgentOperations:
    """Namespace for agent operations (accessed as 'agents' in DSL)."""

    @staticmethod
    def alloc(count: int, properties: Dict[str, Any], **kwargs) -> Agents:
        """Allocate a new agent collection.

        Args:
            count: Number of agents to allocate
            properties: Dictionary mapping property names to initial values
                       Values can be:
                       - NumPy arrays (shape[0] must equal count)
                       - Scalars (broadcast to all agents)
                       - Field2D objects (will be converted to arrays)

        Returns:
            New Agents instance

        Example:
            agents = agents.alloc(
                count=100,
                properties={
                    'pos': np.random.rand(100, 2),
                    'vel': np.zeros((100, 2)),
                    'mass': 1.0  # Broadcast to all agents
                }
            )
        """
        processed_props = {}

        for name, value in properties.items():
            if isinstance(value, np.ndarray):
                # Already an array
                if value.shape[0] != count:
                    raise ValueError(
                        f"Property '{name}' array has {value.shape[0]} elements, "
                        f"expected {count}"
                    )
                processed_props[name] = value.copy()

            elif np.isscalar(value):
                # Broadcast scalar to all agents
                processed_props[name] = np.full(count, value, dtype=np.float32)

            else:
                # Try to convert to array
                try:
                    arr = np.array(value, dtype=np.float32)
                    if arr.shape[0] != count:
                        # Broadcast if needed
                        arr = np.full(count, arr, dtype=np.float32)
                    processed_props[name] = arr
                except Exception as e:
                    raise TypeError(
                        f"Cannot convert property '{name}' to array: {e}"
                    )

        return Agents(count=count, properties=processed_props)

    @staticmethod
    def map(agents_obj: Agents, property_name: str, func: Callable) -> np.ndarray:
        """Apply function to each agent's property.

        Args:
            agents_obj: Agents collection
            property_name: Property to map over
            func: Function to apply element-wise

        Returns:
            Array of mapped values

        Example:
            # Move all agents right by 1.0
            new_pos = agents.map(agents_obj, 'pos', lambda p: p + np.array([1.0, 0.0]))
        """
        values = agents_obj.get(property_name)

        # For vectorized operations
        if callable(func):
            try:
                # Try vectorized operation first
                result = func(values)
                return result
            except Exception:
                # Fall back to element-wise if vectorization fails
                return np.array([func(v) for v in values])
        else:
            raise TypeError(f"Expected callable, got {type(func)}")

    @staticmethod
    def filter(agents_obj: Agents, property_name: str, condition: Callable) -> Agents:
        """Keep only agents matching condition.

        Args:
            agents_obj: Agents collection
            property_name: Property to test
            condition: Function that returns bool for each value

        Returns:
            New Agents instance with filtered alive_mask

        Example:
            # Keep only agents with positive x position
            filtered = agents.filter(agents_obj, 'pos', lambda p: p[0] > 0.0)
        """
        values = agents_obj.get_all(property_name)

        # Apply condition to get mask
        try:
            # Try vectorized first
            mask = condition(values)
        except Exception:
            # Fall back to element-wise
            mask = np.array([condition(v) for v in values], dtype=bool)

        # Create new agents with updated mask
        new_agents = agents_obj.copy()
        new_agents.alive_mask = agents_obj.alive_mask & mask
        return new_agents

    @staticmethod
    def reduce(agents_obj: Agents, property_name: str,
               operation: str = "sum", initial: Optional[Any] = None) -> Any:
        """Reduce agents to single value.

        Args:
            agents_obj: Agents collection
            property_name: Property to reduce
            operation: Reduction operation ("sum", "mean", "min", "max", "prod")
            initial: Initial value (not used for built-in operations)

        Returns:
            Reduced value

        Example:
            # Total mass of all agents
            total_mass = agents.reduce(agents_obj, 'mass', operation='sum')
        """
        values = agents_obj.get(property_name)

        if operation == "sum":
            return np.sum(values)
        elif operation == "mean":
            return np.mean(values)
        elif operation == "min":
            return np.min(values)
        elif operation == "max":
            return np.max(values)
        elif operation == "prod":
            return np.prod(values)
        else:
            raise ValueError(f"Unknown reduction operation: {operation}")

    @staticmethod
    def compute_pairwise_forces(
        agents_obj: Agents,
        radius: float,
        force_func: Callable,
        position_property: str = 'pos',
        mass_property: Optional[str] = None,
        use_spatial_hashing: bool = True
    ) -> np.ndarray:
        """Compute forces between nearby agents.

        Uses spatial hashing for O(n) performance when use_spatial_hashing=True,
        otherwise falls back to O(n²) brute force.

        Args:
            agents_obj: Agents collection
            radius: Interaction radius
            force_func: Function(pos_i, pos_j, [mass_i, mass_j]) -> force_vector
            position_property: Name of position property
            mass_property: Optional name of mass property
            use_spatial_hashing: Whether to use spatial hashing optimization

        Returns:
            Array of force vectors for each agent

        Example:
            # Gravitational forces
            forces = agents.compute_pairwise_forces(
                agents_obj,
                radius=100.0,
                force_func=lambda pi, pj, mi, mj: compute_gravity(pi, pj, mi, mj)
            )
        """
        positions = agents_obj.get(position_property)
        n_agents = len(positions)

        # Determine dimensionality
        if len(positions.shape) == 1:
            dim = 1
            forces = np.zeros(n_agents, dtype=np.float32)
        else:
            dim = positions.shape[1]
            forces = np.zeros((n_agents, dim), dtype=np.float32)

        # Get masses if provided
        masses = None
        if mass_property is not None:
            masses = agents_obj.get(mass_property)

        if use_spatial_hashing and dim >= 2:
            # Use spatial hashing for 2D/3D
            forces = AgentOperations._pairwise_forces_spatial_hash(
                positions, radius, force_func, masses, dim
            )
        else:
            # Brute force O(n²) for small counts or 1D
            forces = AgentOperations._pairwise_forces_brute(
                positions, radius, force_func, masses, dim
            )

        return forces

    @staticmethod
    def _pairwise_forces_brute(
        positions: np.ndarray,
        radius: float,
        force_func: Callable,
        masses: Optional[np.ndarray],
        dim: int
    ) -> np.ndarray:
        """Brute force O(n²) pairwise force calculation."""
        n_agents = len(positions)
        forces = np.zeros_like(positions, dtype=np.float32)

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Compute distance
                if dim == 1:
                    delta = positions[j] - positions[i]
                    dist = abs(delta)
                else:
                    delta = positions[j] - positions[i]
                    dist = np.linalg.norm(delta)

                # Skip if too far
                if dist > radius:
                    continue

                # Compute force
                if masses is not None:
                    force = force_func(positions[i], positions[j], masses[i], masses[j])
                else:
                    force = force_func(positions[i], positions[j])

                # Newton's third law
                forces[i] += force
                forces[j] -= force

        return forces

    @staticmethod
    def _pairwise_forces_spatial_hash(
        positions: np.ndarray,
        radius: float,
        force_func: Callable,
        masses: Optional[np.ndarray],
        dim: int
    ) -> np.ndarray:
        """Spatial hashing O(n) pairwise force calculation."""
        n_agents = len(positions)
        forces = np.zeros_like(positions, dtype=np.float32)

        # Build spatial hash grid
        cell_size = radius
        grid = {}

        for i, pos in enumerate(positions):
            # Compute cell coordinates
            if dim == 2:
                cell = (int(pos[0] / cell_size), int(pos[1] / cell_size))
            else:  # dim == 3
                cell = (int(pos[0] / cell_size), int(pos[1] / cell_size),
                       int(pos[2] / cell_size))

            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        # For each agent, check neighboring cells
        neighbor_offsets = []
        if dim == 2:
            neighbor_offsets = [
                (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            ]
        else:  # dim == 3
            neighbor_offsets = [
                (dx, dy, dz)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
            ]

        for i, pos_i in enumerate(positions):
            # Get agent's cell
            if dim == 2:
                cell = (int(pos_i[0] / cell_size), int(pos_i[1] / cell_size))
            else:
                cell = (int(pos_i[0] / cell_size), int(pos_i[1] / cell_size),
                       int(pos_i[2] / cell_size))

            # Check neighboring cells
            for offset in neighbor_offsets:
                neighbor_cell = tuple(c + o for c, o in zip(cell, offset))

                if neighbor_cell not in grid:
                    continue

                for j in grid[neighbor_cell]:
                    if j <= i:  # Skip self and already-processed pairs
                        continue

                    # Compute distance
                    delta = positions[j] - pos_i
                    dist = np.linalg.norm(delta)

                    if dist > radius:
                        continue

                    # Compute force
                    if masses is not None:
                        force = force_func(pos_i, positions[j], masses[i], masses[j])
                    else:
                        force = force_func(pos_i, positions[j])

                    # Newton's third law
                    forces[i] += force
                    forces[j] -= force

        return forces

    @staticmethod
    def sample_field(agents_obj: Agents, field, position_property: str = 'pos') -> np.ndarray:
        """Sample field values at agent positions.

        Uses bilinear interpolation for 2D fields.

        Args:
            agents_obj: Agents collection
            field: Field2D object to sample from
            position_property: Name of position property

        Returns:
            Array of sampled field values at each agent position

        Example:
            # Sample temperature at each agent
            temps = agents.sample_field(agents_obj, temperature_field, 'pos')
        """
        from .field import Field2D

        if not isinstance(field, Field2D):
            raise TypeError(f"Expected Field2D, got {type(field)}")

        positions = agents_obj.get(position_property)

        # Positions are in world coordinates, need to map to grid coordinates
        # Assume field covers [0, width) x [0, height) in world coords
        h, w = field.shape

        # Clamp positions to field bounds
        x = np.clip(positions[:, 0], 0, w - 1)
        y = np.clip(positions[:, 1], 0, h - 1)

        # Bilinear interpolation
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = x - x0
        fy = y - y0

        # Sample field at corners
        if len(field.data.shape) == 2:
            # Scalar field
            sampled = (
                field.data[y0, x0] * (1 - fx) * (1 - fy) +
                field.data[y0, x1] * fx * (1 - fy) +
                field.data[y1, x0] * (1 - fx) * fy +
                field.data[y1, x1] * fx * fy
            )
        else:
            # Vector field - sample each channel
            n_channels = field.data.shape[2]
            sampled = np.zeros((len(positions), n_channels), dtype=np.float32)

            for c in range(n_channels):
                sampled[:, c] = (
                    field.data[y0, x0, c] * (1 - fx) * (1 - fy) +
                    field.data[y0, x1, c] * fx * (1 - fy) +
                    field.data[y1, x0, c] * (1 - fx) * fy +
                    field.data[y1, x1, c] * fx * fy
                )

        return sampled


# Create singleton instance for use as 'agents' namespace
agents = AgentOperations()
