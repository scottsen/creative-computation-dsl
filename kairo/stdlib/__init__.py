"""Standard library implementations for Creative Computation DSL."""

# Core domains
from .field import field, Field2D, FieldOperations
from .visual import visual, Visual, VisualOperations

# Base-level domains
from . import integrators
from . import io_storage
from . import sparse_linalg

# Procedural graphics domains (NEW)
from .noise import noise, NoiseField2D, NoiseField3D, NoiseOperations
from .palette import palette, Palette, PaletteOperations
from .color import color, ColorOperations
from .image import image, Image, ImageOperations

__all__ = [
    # Core domains
    "field", "Field2D", "FieldOperations",
    "visual", "Visual", "VisualOperations",

    # Base-level domains
    "integrators",
    "io_storage",
    "sparse_linalg",

    # Procedural graphics domains
    "noise", "NoiseField2D", "NoiseField3D", "NoiseOperations",
    "palette", "Palette", "PaletteOperations",
    "color", "ColorOperations",
    "image", "Image", "ImageOperations",
]
