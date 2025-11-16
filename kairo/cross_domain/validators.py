"""
Cross-Domain Type Validation

Validates type compatibility and data flow correctness across domain boundaries.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class CrossDomainTypeError(TypeError):
    """Raised when types are incompatible across domain boundaries."""

    def __init__(self, source_domain: str, target_domain: str, message: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        super().__init__(
            f"Cross-domain type error ({source_domain} → {target_domain}): {message}"
        )


class CrossDomainValidationError(ValueError):
    """Raised when cross-domain flow validation fails."""

    def __init__(self, source_domain: str, target_domain: str, message: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        super().__init__(
            f"Cross-domain validation error ({source_domain} → {target_domain}): {message}"
        )


def validate_cross_domain_flow(
    source_domain: str,
    target_domain: str,
    source_data: Any,
    interface_class: Optional[Any] = None
) -> bool:
    """
    Validate a cross-domain data flow.

    Args:
        source_domain: Source domain name
        target_domain: Target domain name
        source_data: Data from source domain
        interface_class: Optional DomainInterface class to use

    Returns:
        True if flow is valid

    Raises:
        CrossDomainTypeError: If types are incompatible
        CrossDomainValidationError: If validation fails
    """
    # If interface_class not provided, look it up
    if interface_class is None:
        from .registry import CrossDomainRegistry
        interface_class = CrossDomainRegistry.get(source_domain, target_domain)

    # Create interface instance
    interface = interface_class(source_data=source_data)

    # Run validation
    try:
        return interface.validate()
    except TypeError as e:
        raise CrossDomainTypeError(source_domain, target_domain, str(e))
    except ValueError as e:
        raise CrossDomainValidationError(source_domain, target_domain, str(e))


def validate_field_data(data: Any, allow_vector: bool = True) -> bool:
    """
    Validate field domain data.

    Args:
        data: Field data (numpy array or object with .data attribute)
        allow_vector: If True, allow multi-channel fields

    Returns:
        True if valid

    Raises:
        TypeError: If data is not a valid field
    """
    # Extract numpy array
    if hasattr(data, 'data'):
        arr = data.data
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError(f"Field data must be numpy array or have .data attribute, got {type(data)}")

    # Check dimensions
    if arr.ndim < 2:
        raise TypeError(f"Field must be at least 2D, got {arr.ndim}D")

    if not allow_vector and arr.ndim > 2:
        raise TypeError(f"Scalar field required, got {arr.ndim}D array")

    return True


def validate_agent_positions(positions: np.ndarray, ndim: int = 2) -> bool:
    """
    Validate agent positions array.

    Args:
        positions: Nx2 or Nx3 array of agent positions
        ndim: Expected spatial dimensions (2 or 3)

    Returns:
        True if valid

    Raises:
        TypeError: If positions are invalid
    """
    if not isinstance(positions, np.ndarray):
        raise TypeError(f"Agent positions must be numpy array, got {type(positions)}")

    if positions.ndim != 2:
        raise TypeError(f"Agent positions must be 2D (N x {ndim}), got shape {positions.shape}")

    if positions.shape[1] != ndim:
        raise TypeError(
            f"Agent positions must be N x {ndim}, got shape {positions.shape}"
        )

    return True


def validate_audio_params(params: Dict[str, np.ndarray]) -> bool:
    """
    Validate audio parameter dictionary.

    Args:
        params: Dict with keys like 'triggers', 'amplitudes', 'frequencies'

    Returns:
        True if valid

    Raises:
        TypeError: If params are invalid
    """
    required_keys = ['triggers']
    optional_keys = ['amplitudes', 'frequencies', 'positions', 'durations']

    # Check required keys
    for key in required_keys:
        if key not in params:
            raise TypeError(f"Audio params missing required key: {key}")

    # Validate array types
    for key, value in params.items():
        if key not in required_keys + optional_keys:
            raise TypeError(f"Unknown audio parameter: {key}")

        if not isinstance(value, (list, np.ndarray)):
            raise TypeError(f"Audio param '{key}' must be list or array, got {type(value)}")

    return True


def check_dimensional_compatibility(
    field_shape: tuple,
    positions: np.ndarray
) -> bool:
    """
    Check if field shape and agent positions are compatible.

    Args:
        field_shape: Shape of field (H, W) or (H, W, C)
        positions: Agent positions array

    Returns:
        True if compatible

    Raises:
        ValueError: If dimensions don't match
    """
    ndim = positions.shape[1]  # Spatial dimensions from positions

    if len(field_shape) < 2:
        raise ValueError(f"Field shape must be at least 2D, got {field_shape}")

    # Check field spatial dims match position dims
    field_spatial_dims = 2 if len(field_shape) <= 3 else 3

    if ndim != field_spatial_dims:
        raise ValueError(
            f"Field is {field_spatial_dims}D but positions are {ndim}D"
        )

    return True


def validate_mapping(
    mapping: Dict[str, str],
    valid_source_props: List[str],
    valid_target_params: List[str]
) -> bool:
    """
    Validate a property mapping dict.

    Args:
        mapping: Dict mapping source properties to target parameters
        valid_source_props: List of valid source property names
        valid_target_params: List of valid target parameter names

    Returns:
        True if valid

    Raises:
        ValueError: If mapping contains invalid properties
    """
    for source_prop, target_param in mapping.items():
        if source_prop not in valid_source_props:
            raise ValueError(
                f"Invalid source property '{source_prop}'. "
                f"Valid options: {valid_source_props}"
            )

        if target_param not in valid_target_params:
            raise ValueError(
                f"Invalid target parameter '{target_param}'. "
                f"Valid options: {valid_target_params}"
            )

    return True
