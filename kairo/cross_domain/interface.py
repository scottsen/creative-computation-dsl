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


class AudioToVisualInterface(DomainInterface):
    """
    Audio → Visual: Audio-reactive visual generation.

    Use cases:
    - FFT spectrum → color palette
    - Amplitude → particle emission
    - Beat detection → visual effects
    - Frequency analysis → color shifts
    """

    source_domain = "audio"
    target_domain = "visual"

    def __init__(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        mode: str = "spectrum"
    ):
        """
        Args:
            audio_signal: Audio samples (mono or stereo)
            sample_rate: Audio sample rate
            fft_size: FFT window size for spectral analysis
            mode: Analysis mode ("spectrum", "waveform", "energy", "beat")
        """
        super().__init__(source_data=audio_signal)
        self.audio_signal = audio_signal
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.mode = mode

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """
        Convert audio to visual parameters.

        Returns:
            Dict with keys: 'colors', 'intensities', 'frequencies', 'energy'
        """
        audio = source_data if source_data is not None else self.audio_signal

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        result = {}

        if self.mode == "spectrum":
            # FFT analysis
            fft = np.fft.rfft(audio[:self.fft_size])
            spectrum = np.abs(fft)

            # Normalize spectrum
            spectrum = spectrum / (np.max(spectrum) + 1e-10)

            result['spectrum'] = spectrum
            result['frequencies'] = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)

            # Spectral centroid (brightness)
            spectral_centroid = np.sum(result['frequencies'] * spectrum) / (np.sum(spectrum) + 1e-10)
            result['brightness'] = spectral_centroid / (self.sample_rate / 2)  # Normalize

        elif self.mode == "waveform":
            # Raw waveform for oscilloscope-style visuals
            result['waveform'] = audio[:self.fft_size]
            result['amplitude'] = np.abs(audio[:self.fft_size])

        elif self.mode == "energy":
            # RMS energy
            energy = np.sqrt(np.mean(audio[:self.fft_size] ** 2))
            result['energy'] = energy
            result['intensity'] = np.clip(energy * 10.0, 0.0, 1.0)

        elif self.mode == "beat":
            # Simple beat detection (onset strength)
            hop_length = 512
            n_frames = len(audio) // hop_length

            onset_strength = []
            for i in range(n_frames):
                start = i * hop_length
                end = start + hop_length
                chunk = audio[start:end]
                energy = np.sqrt(np.mean(chunk ** 2))
                onset_strength.append(energy)

            onset_strength = np.array(onset_strength)

            # Detect peaks
            threshold = np.mean(onset_strength) + np.std(onset_strength)
            beats = onset_strength > threshold

            result['onset_strength'] = onset_strength
            result['beats'] = beats

        return result

    def validate(self) -> bool:
        """Check audio signal is valid."""
        if self.audio_signal is None:
            return False

        if not isinstance(self.audio_signal, np.ndarray):
            raise TypeError("Audio signal must be numpy array")

        if len(self.audio_signal) < self.fft_size:
            raise ValueError(f"Audio signal too short (need at least {self.fft_size} samples)")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'audio_signal': np.ndarray,
            'sample_rate': int,
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'visual_params': Dict[str, np.ndarray],
        }


class FieldToAudioInterface(DomainInterface):
    """
    Field → Audio: Field-driven audio synthesis.

    Use cases:
    - Temperature field → synthesis parameters
    - Vorticity → frequency modulation
    - Density patterns → rhythm generation
    - Field evolution → audio sequences
    """

    source_domain = "field"
    target_domain = "audio"

    def __init__(
        self,
        field: np.ndarray,
        mapping: Dict[str, str],
        sample_rate: int = 44100,
        duration: float = 1.0
    ):
        """
        Args:
            field: 2D field array
            mapping: Dict mapping field properties to audio parameters
                     e.g., {"mean": "frequency", "std": "amplitude"}
            sample_rate: Audio sample rate
            duration: Duration of generated audio (seconds)
        """
        super().__init__(source_data=field)
        self.field = field
        self.mapping = mapping
        self.sample_rate = sample_rate
        self.duration = duration

    def transform(self, source_data: Any) -> Dict[str, Any]:
        """
        Convert field to audio synthesis parameters.

        Returns:
            Dict with synthesis parameters
        """
        field = source_data if source_data is not None else self.field

        # Extract field statistics
        stats = {
            'mean': np.mean(field),
            'std': np.std(field),
            'min': np.min(field),
            'max': np.max(field),
            'range': np.ptp(field),
        }

        # Compute spatial statistics
        if field.ndim >= 2:
            # Gradient magnitude (activity/turbulence)
            gy, gx = np.gradient(field)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            stats['gradient_mean'] = np.mean(gradient_mag)
            stats['gradient_max'] = np.max(gradient_mag)

        audio_params = {}

        # Map field properties to audio parameters
        for field_prop, audio_param in self.mapping.items():
            value = stats.get(field_prop, 0.0)

            if audio_param == "frequency":
                # Map to musical frequency range (100-1000 Hz)
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['frequency'] = 100.0 + normalized * 900.0

            elif audio_param == "amplitude":
                # Map to amplitude (0-1)
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['amplitude'] = np.clip(normalized, 0.0, 1.0)

            elif audio_param == "modulation":
                # Modulation depth
                audio_params['modulation_depth'] = np.clip(value / 10.0, 0.0, 1.0)

            elif audio_param == "filter_cutoff":
                # Filter cutoff frequency
                normalized = (value - stats['min']) / (stats['range'] + 1e-10)
                audio_params['filter_cutoff'] = 200.0 + normalized * 3800.0

        # Add timing info
        audio_params['sample_rate'] = self.sample_rate
        audio_params['duration'] = self.duration
        audio_params['n_samples'] = int(self.sample_rate * self.duration)

        return audio_params

    def validate(self) -> bool:
        """Check field and mapping are valid."""
        if self.field is None or self.mapping is None:
            return False

        valid_field_props = ['mean', 'std', 'min', 'max', 'range', 'gradient_mean', 'gradient_max']
        valid_audio_params = ['frequency', 'amplitude', 'modulation', 'filter_cutoff']

        for field_prop, audio_param in self.mapping.items():
            if field_prop not in valid_field_props:
                raise ValueError(f"Unknown field property: {field_prop}")
            if audio_param not in valid_audio_params:
                raise ValueError(f"Unknown audio parameter: {audio_param}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {
            'field': np.ndarray,
            'mapping': Dict[str, str],
        }

    def get_output_interface(self) -> Dict[str, Type]:
        return {
            'audio_params': Dict[str, Any],
        }


class TerrainToFieldInterface(DomainInterface):
    """
    Terrain → Field: Convert terrain heightmap to scalar field.

    Use cases:
    - Heightmap → diffusion initial conditions
    - Elevation → potential field
    - Terrain features → field patterns
    """

    source_domain = "terrain"
    target_domain = "field"

    def __init__(self, heightmap: np.ndarray, normalize: bool = True):
        """
        Args:
            heightmap: 2D terrain heightmap
            normalize: If True, normalize to [0, 1] range
        """
        super().__init__(source_data=heightmap)
        self.heightmap = heightmap
        self.normalize = normalize

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert heightmap to field."""
        heightmap = source_data if source_data is not None else self.heightmap

        # Extract height data if wrapped in object
        if hasattr(heightmap, 'data'):
            field_data = heightmap.data.copy()
        else:
            field_data = heightmap.copy()

        if self.normalize:
            # Normalize to [0, 1]
            field_min = field_data.min()
            field_max = field_data.max()
            if field_max > field_min:
                field_data = (field_data - field_min) / (field_max - field_min)

        return field_data

    def validate(self) -> bool:
        """Check heightmap is valid."""
        if self.heightmap is None:
            return False

        # Extract array
        if hasattr(self.heightmap, 'data'):
            arr = self.heightmap.data
        else:
            arr = self.heightmap

        if not isinstance(arr, np.ndarray):
            raise TypeError("Heightmap must be numpy array")

        if arr.ndim != 2:
            raise ValueError(f"Heightmap must be 2D, got shape {arr.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'heightmap': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}


class FieldToTerrainInterface(DomainInterface):
    """
    Field → Terrain: Convert scalar field to terrain heightmap.

    Use cases:
    - Procedural field → terrain generation
    - Simulation result → landscape
    """

    source_domain = "field"
    target_domain = "terrain"

    def __init__(self, field: np.ndarray, height_scale: float = 100.0):
        """
        Args:
            field: 2D scalar field
            height_scale: Scaling factor for height values
        """
        super().__init__(source_data=field)
        self.field = field
        self.height_scale = height_scale

    def transform(self, source_data: Any) -> Dict[str, np.ndarray]:
        """Convert field to heightmap."""
        field = source_data if source_data is not None else self.field

        # Normalize field to [0, 1]
        field_min = field.min()
        field_max = field.max()
        normalized = (field - field_min) / (field_max - field_min + 1e-10)

        # Scale to height range
        heightmap = normalized * self.height_scale

        return {
            'heightmap': heightmap,
            'min_height': 0.0,
            'max_height': self.height_scale,
        }

    def validate(self) -> bool:
        """Check field is valid."""
        if self.field is None:
            return False

        if not isinstance(self.field, np.ndarray):
            raise TypeError("Field must be numpy array")

        if self.field.ndim != 2:
            raise ValueError(f"Field must be 2D, got shape {self.field.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'terrain_data': Dict[str, np.ndarray]}


class VisionToFieldInterface(DomainInterface):
    """
    Vision → Field: Convert computer vision features to fields.

    Use cases:
    - Edge map → scalar field
    - Optical flow → vector field
    - Feature map → field initialization
    """

    source_domain = "vision"
    target_domain = "field"

    def __init__(self, image: np.ndarray, mode: str = "edges"):
        """
        Args:
            image: Input image (grayscale or RGB)
            mode: Conversion mode ("edges", "gradient", "intensity")
        """
        super().__init__(source_data=image)
        self.image = image
        self.mode = mode

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert vision data to field."""
        image = source_data if source_data is not None else self.image

        # Convert to grayscale if RGB
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        if self.mode == "edges":
            # Edge detection produces scalar field
            from scipy.ndimage import sobel
            sx = sobel(image, axis=1)
            sy = sobel(image, axis=0)
            edge_mag = np.sqrt(sx**2 + sy**2)
            return edge_mag

        elif self.mode == "gradient":
            # Gradient field (vector)
            gy, gx = np.gradient(image)
            # Return as vector field (H, W, 2)
            return np.stack([gy, gx], axis=2)

        elif self.mode == "intensity":
            # Direct intensity mapping
            return image.astype(np.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def validate(self) -> bool:
        """Check image is valid."""
        if self.image is None:
            return False

        if not isinstance(self.image, np.ndarray):
            raise TypeError("Image must be numpy array")

        if self.image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got shape {self.image.shape}")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'image': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}


class GraphToVisualInterface(DomainInterface):
    """
    Graph → Visual: Network graph visualization.

    Use cases:
    - Network structure → visual layout
    - Graph metrics → node colors/sizes
    - Connectivity → edge rendering
    """

    source_domain = "graph"
    target_domain = "visual"

    def __init__(
        self,
        graph_data: Dict[str, Any],
        width: int = 512,
        height: int = 512,
        layout: str = "spring"
    ):
        """
        Args:
            graph_data: Dict with 'nodes' and 'edges' keys
            width: Output image width
            height: Output image height
            layout: Layout algorithm ("spring", "circular", "random")
        """
        super().__init__(source_data=graph_data)
        self.graph_data = graph_data
        self.width = width
        self.height = height
        self.layout = layout

    def transform(self, source_data: Any) -> Dict[str, Any]:
        """
        Convert graph to visual representation.

        Returns:
            Dict with 'node_positions', 'edge_list', 'image' keys
        """
        graph = source_data if source_data is not None else self.graph_data

        n_nodes = len(graph.get('nodes', []))

        # Simple layout algorithms
        if self.layout == "circular":
            # Circular layout
            angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            radius = min(self.width, self.height) * 0.4
            cx, cy = self.width / 2, self.height / 2

            positions = np.zeros((n_nodes, 2))
            positions[:, 0] = cx + radius * np.cos(angles)
            positions[:, 1] = cy + radius * np.sin(angles)

        elif self.layout == "random":
            # Random layout
            positions = np.random.rand(n_nodes, 2)
            positions[:, 0] *= self.width
            positions[:, 1] *= self.height

        elif self.layout == "spring":
            # Simple spring layout (simplified force-directed)
            positions = np.random.rand(n_nodes, 2)
            positions[:, 0] *= self.width
            positions[:, 1] *= self.height

            # Simple relaxation (could be improved with proper spring algorithm)
            for _ in range(50):
                # Repulsion between all nodes
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        delta = positions[i] - positions[j]
                        dist = np.linalg.norm(delta) + 1e-10
                        force = delta / dist * (100.0 / dist)
                        positions[i] += force * 0.1
                        positions[j] -= force * 0.1

                # Clamp to bounds
                positions[:, 0] = np.clip(positions[:, 0], 0, self.width)
                positions[:, 1] = np.clip(positions[:, 1], 0, self.height)

        return {
            'node_positions': positions,
            'edge_list': graph.get('edges', []),
            'n_nodes': n_nodes,
            'width': self.width,
            'height': self.height,
        }

    def validate(self) -> bool:
        """Check graph data is valid."""
        if self.graph_data is None:
            return False

        if 'nodes' not in self.graph_data:
            raise ValueError("Graph data must have 'nodes' key")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'graph_data': Dict}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'visual_data': Dict[str, Any]}


class CellularToFieldInterface(DomainInterface):
    """
    Cellular → Field: Convert cellular automata state to field.

    Use cases:
    - CA state → initial conditions for PDEs
    - Game of Life → density field
    - Pattern state → field patterns
    """

    source_domain = "cellular"
    target_domain = "field"

    def __init__(self, ca_state: np.ndarray, normalize: bool = True):
        """
        Args:
            ca_state: Cellular automata state array
            normalize: If True, normalize to [0, 1]
        """
        super().__init__(source_data=ca_state)
        self.ca_state = ca_state
        self.normalize = normalize

    def transform(self, source_data: Any) -> np.ndarray:
        """Convert CA state to field."""
        ca_state = source_data if source_data is not None else self.ca_state

        field = ca_state.astype(np.float32)

        if self.normalize:
            field_min = field.min()
            field_max = field.max()
            if field_max > field_min:
                field = (field - field_min) / (field_max - field_min)

        return field

    def validate(self) -> bool:
        """Check CA state is valid."""
        if self.ca_state is None:
            return False

        if not isinstance(self.ca_state, np.ndarray):
            raise TypeError("CA state must be numpy array")

        return True

    def get_input_interface(self) -> Dict[str, Type]:
        return {'ca_state': np.ndarray}

    def get_output_interface(self) -> Dict[str, Type]:
        return {'field': np.ndarray}
