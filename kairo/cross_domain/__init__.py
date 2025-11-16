"""
Cross-Domain Integration for Kairo

This module provides the infrastructure for composing operators across different
computational domains (Field, Agent, Audio, Physics, Geometry, etc.).

Key components:
- DomainInterface: Base class for inter-domain data flows
- Transform functions: Convert data between domain formats
- Type validation: Ensure compatibility across domain boundaries
- Composition operators: compose() and link() support
"""

from .interface import DomainInterface, DomainTransform
from .registry import CrossDomainRegistry, register_transform
from .validators import validate_cross_domain_flow, CrossDomainTypeError

__all__ = [
    'DomainInterface',
    'DomainTransform',
    'CrossDomainRegistry',
    'register_transform',
    'validate_cross_domain_flow',
    'CrossDomainTypeError',
]
