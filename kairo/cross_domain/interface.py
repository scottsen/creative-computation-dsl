"""
Domain Interface Base Classes

Provides the foundational abstractions for cross-domain data flows in Kairo.
Based on ADR-002: Cross-Domain Architectural Patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from dataclasses import dataclass
import numpy as np


@dataclass
class DomainMetadata:
    """Metadata describing a domain's capabilities and interfaces."""

    name: str
    version: str
    input_types: Set[str]  # What types this domain can accept
    output_types: Set[str]  # What types this domain can provide
    dependencies: List[str]  # Other domains this depends on
    description: str


class DomainInterface(ABC):
    """
    Base class for inter-domain data flows.

    Each domain pair (source → target) that supports composition must implement
    a DomainInterface subclass that handles:
    1. Type validation
    2. Data transformation
    3. Metadata propagation

    Example:
        class FieldToAgentInterface(DomainInterface):
            source_domain = "field"
            target_domain = "agent"

            def transform(self, field_data):
                # Sample field at agent positions
                return sampled_values

            def validate(self):
                # Check field dimensions, agent count, etc.
                return True
    """

    source_domain: str = None  # Set by subclass
    target_domain: str = None  # Set by subclass

    def __init__(self, source_data: Any = None, metadata: Optional[Dict] = None):
        self.source_data = source_data
        self.metadata = metadata or {}
        self._validated = False

    @abstractmethod
    def transform(self, source_data: Any) -> Any:
        """
        Convert source domain data to target domain format.

        Args:
            source_data: Data in source domain format

        Returns:
            Data in target domain format

        Raises:
            ValueError: If data cannot be transformed
            TypeError: If data types are incompatible
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform()"
        )

    @abstractmethod
    def validate(self) -> bool:
        """
        Ensure data types are compatible across domains.

        Returns:
            True if transformation is valid, False otherwise

        Raises:
            CrossDomainTypeError: If types are fundamentally incompatible
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )

    def get_input_interface(self) -> Dict[str, Type]:
        """
        Describe what data this interface can accept.

        Returns:
            Dict mapping parameter names to their types
        """
        return {}

    def get_output_interface(self) -> Dict[str, Type]:
        """
        Describe what data this interface can provide.

        Returns:
            Dict mapping output names to their types
        """
        return {}

    def __call__(self, source_data: Any) -> Any:
        """
        Convenience method: validate and transform in one call.

        Args:
            source_data: Data to transform

        Returns:
            Transformed data
        """
        self.source_data = source_data
        if not self._validated:
            if not self.validate():
                raise ValueError(
                    f"Cross-domain flow {self.source_domain} → {self.target_domain} "
                    f"failed validation"
                )
            self._validated = True

        return self.transform(source_data)


class DomainTransform:
    """
    Decorator for registering cross-domain transform functions.

    Example:
        @DomainTransform(source="field", target="agent")
        def field_to_agent_force(field, agent_positions):
            '''Sample field values at agent positions.'''
            return sample_field(field, agent_positions)
    """

    def __init__(
        self,
        source: str,
        target: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_types: Optional[Dict[str, Type]] = None,
        output_type: Optional[Type] = None,
    ):
        self.source = source
        self.target = target
        self.name = name
        self.description = description
        self.input_types = input_types or {}
        self.output_type = output_type
        self.transform_fn = None

    def __call__(self, fn):
        """Register the decorated function as a transform."""
        self.transform_fn = fn
        self.name = self.name or fn.__name__
        self.description = self.description or fn.__doc__

        # Create a DomainInterface wrapper
        class TransformInterface(DomainInterface):
            source_domain = self.source
            target_domain = self.target

            def transform(iself, source_data: Any) -> Any:
                return fn(source_data)

            def validate(iself) -> bool:
                # Basic type checking if types specified
                if self.input_types:
                    # TODO: Implement type validation
                    pass
                return True

        # Store metadata
        TransformInterface.__name__ = f"{self.source}To{self.target.capitalize()}Transform"
        TransformInterface.__doc__ = self.description

        # Register in global registry (will be implemented)
        from .registry import CrossDomainRegistry
        CrossDomainRegistry.register(self.source, self.target, TransformInterface)

        return fn


# ============================================================================
# Concrete Domain Interfaces
# ============================================================================


class FieldToAgentInterface(DomainInterface):
    """
    Field → Agent: Sample field values at agent positions.

    Use cases:
    - Flow field → force on particles
    - Temperature field → agent color/behavior
    - Density field → agent sensing
    """

    source_domain = "field"
    target_domain = "agent"

    def __init__(self, field, positions, property_name="value"):
        super().__init__(source_data=field)
        self.field = field
        self.positions = positions
        self.property_name = property_name

    def transform(self, source_data: Any) -> np.ndarray:
        """Sample field at agent positions."""
        field = source_data if source_data is not None else self.field

        # Handle different field types
        if hasattr(field, 'data'):
            field_data = field.data
        elif isinstance(field, np.ndarray):
            field_data = field
        else:
            raise TypeError(f"Unknown field type: {type(field)}")

        # Sample using bilinear interpolation
        sampled = self._sample_field(field_data, self.positions)
        return sampled

    def validate(self) -> bool:
        """Check field and positions are compatible."""
        if self.field is None or self.positions is None:
            return False

        # Check positions are 2D (Nx2)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError(
                f"Agent positions must be Nx2, got shape {self.positions.shape}"
            )

        return True

    def _sample_field(self, field_data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Sample field at positions using bilinear interpolation.

        Args:
            field_data: 2D or 3D array (H, W) or (H, W, C)
            positions: Nx2 array of (y, x) coordinates

        Returns:
            N-length array of sampled values (or NxC for vector fields)
        """
        from scipy.ndimage import map_coordinates

        # Ensure field_data is a numpy array (not memoryview)
        field_data = np.asarray(field_data)

        # Normalize positions to grid coordinates
        h, w = field_data.shape[:2]
        coords = positions.copy()

        # Clamp to valid range
        coords[:, 0] = np.clip(coords[:, 0], 0, h - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, w - 1)

        # Sample using scipy map_coordinates
        if field_data.ndim == 2:
            # Scalar field
            sampled = map_coordinates(
                field_data,
                [coords[:, 0], coords[:, 1]],
                order=1,  # Bilinear
                mode='nearest'
            )
        else:
            # Vector field - sample each component
            sampled = np.zeros((len(positions), field_data.shape[2]), dtype=field_data.dtype)
            for c in range(field_data.shape[2]):
                component_data = np.asarray(field_data[:, :, c])
                sampled[:, c] = map_coordinates(
                    component_data,
                    [coords[:, 0], coords[:, 1]],
                    order=1,
                    mode='nearest'
                )

        return sampled

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
            'positions': np.ndarray,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'sampled_values': np.ndarray,
        }


class AgentToFieldInterface(DomainInterface):
    """
    Agent → Field: Deposit agent properties onto field grid.

    Use cases:
    - Particle positions → density field
    - Agent velocities → velocity field
    - Agent properties → heat sources
    """

    source_domain = "agent"
    target_domain = "field"

    def __init__(
        self,
        positions,
        values,
        field_shape: Tuple[int, int],
        method: str = "accumulate"
    ):
        super().__init__(source_data=(positions, values))
        self.positions = positions
        self.values = values
        self.field_shape = field_shape
        self.method = method  # "accumulate", "average", "max"

    def transform(self, source_data: Any) -> np.ndarray:
        """Deposit agent values onto field."""
        if source_data is not None:
            positions, values = source_data
        else:
            positions, values = self.positions, self.values

        field = np.zeros(self.field_shape, dtype=np.float32)

        # Convert positions to grid coordinates
        coords = positions.astype(int)

        # Clamp to valid range
        coords[:, 0] = np.clip(coords[:, 0], 0, self.field_shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, self.field_shape[1] - 1)

        if self.method == "accumulate":
            # Sum all values at each grid cell
            for i, (y, x) in enumerate(coords):
                field[y, x] += values[i]

        elif self.method == "average":
            # Average values at each grid cell
            counts = np.zeros(self.field_shape, dtype=int)
            for i, (y, x) in enumerate(coords):
                field[y, x] += values[i]
                counts[y, x] += 1

            # Avoid division by zero
            mask = counts > 0
            field[mask] /= counts[mask]

        elif self.method == "max":
            # Take maximum value at each grid cell
            field.fill(-np.inf)
            for i, (y, x) in enumerate(coords):
                field[y, x] = max(field[y, x], values[i])
            field[field == -np.inf] = 0

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return field

    def validate(self) -> bool:
        """Check positions and values are compatible."""
        if self.positions is None or self.values is None:
            return False

        if len(self.positions) != len(self.values):
            raise ValueError(
                f"Positions ({len(self.positions)}) and values ({len(self.values)}) "
                f"must have same length"
            )

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'positions': np.ndarray,
            'values': np.ndarray,
            'field_shape': Tuple[int, int],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
        }


class PhysicsToAudioInterface(DomainInterface):
    """
    Physics → Audio: Sonification of physical events.

    Use cases:
    - Collision forces → percussion synthesis
    - Body velocities → pitch/volume
    - Contact points → spatial audio
    """

    source_domain = "physics"
    target_domain = "audio"

    def __init__(
        self,
        events,
        mapping: Dict[str, str],
        sample_rate: int = 48000
    ):
        """
        Args:
            events: Physical events (collisions, contacts, etc.)
            mapping: Dict mapping physics properties to audio parameters
                     e.g., {"impulse": "amplitude", "body_id": "pitch"}
            sample_rate: Audio sample rate
        """
        super().__init__(source_data=events)
        self.events = events
        self.mapping = mapping
        self.sample_rate = sample_rate

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """
        Convert physics events to audio parameters.

        Returns:
            Dict with keys: 'triggers', 'amplitudes', 'frequencies', 'positions'
        """
        events = source_data if source_data is not None else self.events

        audio_params = {
            'triggers': [],
            'amplitudes': [],
            'frequencies': [],
            'positions': [],
        }

        for event in events:
            # Extract physics properties based on mapping
            if "impulse" in self.mapping:
                audio_param = self.mapping["impulse"]
                impulse = getattr(event, "impulse", 1.0)

                if audio_param == "amplitude":
                    # Map impulse magnitude to volume (0-1)
                    amplitude = np.clip(impulse / 100.0, 0.0, 1.0)
                    audio_params['amplitudes'].append(amplitude)

            if "body_id" in self.mapping:
                audio_param = self.mapping["body_id"]
                body_id = getattr(event, "body_id", 0)

                if audio_param == "pitch":
                    # Map body ID to frequency (C major scale)
                    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
                    freq = frequencies[body_id % len(frequencies)]
                    audio_params['frequencies'].append(freq)

            if "position" in self.mapping:
                pos = getattr(event, "position", (0, 0))
                audio_params['positions'].append(pos)

            # Trigger time (in samples)
            trigger_time = getattr(event, "time", 0.0)
            audio_params['triggers'].append(int(trigger_time * self.sample_rate))

        return audio_params

    def validate(self) -> bool:
        """Check events and mapping are valid."""
        if not self.events or not self.mapping:
            return False

        valid_physics_props = ["impulse", "body_id", "position", "velocity", "time"]
        valid_audio_params = ["amplitude", "pitch", "pan", "duration"]

        for phys_prop, audio_param in self.mapping.items():
            if phys_prop not in valid_physics_props:
                raise ValueError(f"Unknown physics property: {phys_prop}")
            if audio_param not in valid_audio_params:
                raise ValueError(f"Unknown audio parameter: {audio_param}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'events': List,
            'mapping': Dict[str, str],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'audio_params': Dict[str, np.ndarray],
        }
