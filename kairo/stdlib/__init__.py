"""Standard library implementations for Creative Computation DSL."""

# Core domains
from .field import field, Field2D, FieldOperations
from .visual import visual, Visual, VisualOperations

# Physics domains
from .acoustics import (
    acoustics, AcousticsOperations, PipeGeometry, WaveguideNetwork,
    ReflectionCoeff, FrequencyResponse, create_pipe, create_expansion_chamber
)
from . import rigidbody

# Base-level domains
from . import integrators
from . import io_storage
from . import sparse_linalg
from .flappy import flappy, Bird, Pipe, GameState, FlappyOperations
from .neural import neural, DenseLayer, MLP, NeuralOperations
from .genetic import genetic, Individual, Population, GeneticOperations

# Agent-based domains
from . import agents
from . import temporal

# Audio domains
from . import audio
from . import signal

# Geometry domains
from . import geometry
from . import graph

# Optimization domains
from . import optimization
from . import statemachine

# Terrain and vision domains
from . import terrain
from . import vision

# Procedural graphics domains
from .noise import noise, NoiseField2D, NoiseField3D, NoiseOperations
from .palette import palette, Palette, PaletteOperations
from .color import color, ColorOperations
from .image import image, Image, ImageOperations
from .cellular import cellular, CellularField2D, CellularField1D, CellularOperations, CARule

__all__ = [
    # Core domains
    "field", "Field2D", "FieldOperations",
    "visual", "Visual", "VisualOperations",

    # Physics domains
    "acoustics", "AcousticsOperations", "PipeGeometry", "WaveguideNetwork",
    "ReflectionCoeff", "FrequencyResponse", "create_pipe", "create_expansion_chamber",
    "rigidbody",

    # Base-level domains
    "integrators",
    "io_storage",
    "sparse_linalg",
    "flappy", "Bird", "Pipe", "GameState", "FlappyOperations",
    "neural", "DenseLayer", "MLP", "NeuralOperations",
    "genetic", "Individual", "Population", "GeneticOperations",

    # Agent-based domains
    "agents",
    "temporal",

    # Audio domains
    "audio",
    "signal",

    # Geometry domains
    "geometry",
    "graph",

    # Optimization domains
    "optimization",
    "statemachine",

    # Terrain and vision domains
    "terrain",
    "vision",

    # Procedural graphics domains
    "noise", "NoiseField2D", "NoiseField3D", "NoiseOperations",
    "palette", "Palette", "PaletteOperations",
    "color", "ColorOperations",
    "image", "Image", "ImageOperations",
    "cellular", "CellularField2D", "CellularField1D", "CellularOperations", "CARule",
]
