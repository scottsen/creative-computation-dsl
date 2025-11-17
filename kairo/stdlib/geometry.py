"""Geometry Domain for 2D/3D geometric primitives and spatial operations.

This module provides geometric primitives, coordinate transformations,
and spatial queries for computational geometry in Kairo.

Features:
- 2D/3D geometric primitives (point, line, circle, rectangle, polygon)
- Coordinate system conversions (Cartesian, polar, spherical)
- Frame-aware transformations (translate, rotate, scale, transform)
- Spatial queries (distance, intersection, containment, closest point)
- Geometric properties (area, perimeter, centroid, bounding box)

Architecture:
- Layer 1: Primitive construction (point, line, circle, rectangle, polygon)
- Layer 2: Transformations (translate, rotate, scale, transform)
- Layer 3: Spatial queries (distance, intersection, contains)
- Layer 4: Coordinate conversions and advanced operations
"""

from typing import Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from kairo.core.operator import operator, OpCategory


# ============================================================================
# CORE TYPES
# ============================================================================


class CoordinateFrame(Enum):
    """Coordinate frame types."""

    CARTESIAN = "cartesian"
    POLAR = "polar"
    SPHERICAL = "spherical"


@dataclass
class Point2D:
    """2D point in Cartesian coordinates.

    Attributes:
        x: X coordinate
        y: Y coordinate
        frame: Coordinate frame (default: Cartesian)
    """

    x: float
    y: float
    frame: CoordinateFrame = CoordinateFrame.CARTESIAN

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array([self.x, self.y])

    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.3f}, y={self.y:.3f})"


@dataclass
class Point3D:
    """3D point in Cartesian coordinates.

    Attributes:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        frame: Coordinate frame (default: Cartesian)
    """

    x: float
    y: float
    z: float
    frame: CoordinateFrame = CoordinateFrame.CARTESIAN

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array([self.x, self.y, self.z])

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


@dataclass
class Line2D:
    """2D line segment defined by two endpoints.

    Attributes:
        start: Start point
        end: End point
    """

    start: Point2D
    end: Point2D

    @property
    def length(self) -> float:
        """Calculate line length."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.sqrt(dx * dx + dy * dy)

    @property
    def direction(self) -> np.ndarray:
        """Get normalized direction vector."""
        vec = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        length = np.linalg.norm(vec)
        return vec / length if length > 0 else np.array([0.0, 0.0])

    def __repr__(self) -> str:
        return f"Line2D({self.start} -> {self.end})"


@dataclass
class Circle:
    """2D circle defined by center and radius.

    Attributes:
        center: Center point
        radius: Circle radius
    """

    center: Point2D
    radius: float

    @property
    def area(self) -> float:
        """Calculate circle area."""
        return np.pi * self.radius * self.radius

    @property
    def circumference(self) -> float:
        """Calculate circle circumference."""
        return 2 * np.pi * self.radius

    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius:.3f})"


@dataclass
class Rectangle:
    """2D axis-aligned rectangle.

    Attributes:
        center: Center point
        width: Rectangle width
        height: Rectangle height
        rotation: Rotation angle in radians (default: 0)
    """

    center: Point2D
    width: float
    height: float
    rotation: float = 0.0

    @property
    def area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)

    def get_vertices(self) -> np.ndarray:
        """Get rectangle vertices in world space.

        Returns:
            Array of 4 vertices [4, 2]
        """
        hw, hh = self.width / 2, self.height / 2

        # Local vertices (centered at origin)
        local_verts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

        # Rotate if needed
        if self.rotation != 0:
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            local_verts = local_verts @ rot_matrix.T

        # Translate to center
        return local_verts + np.array([self.center.x, self.center.y])

    def __repr__(self) -> str:
        return f"Rectangle(center={self.center}, w={self.width:.3f}, h={self.height:.3f})"


@dataclass
class Polygon:
    """2D polygon defined by vertices.

    Attributes:
        vertices: List of vertices (counter-clockwise winding)
    """

    vertices: np.ndarray  # Shape: [N, 2]

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)

    @property
    def area(self) -> float:
        """Calculate polygon area using shoelace formula.

        Returns:
            Signed area (positive for CCW, negative for CW winding)
        """
        n = len(self.vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]

        return abs(area) / 2.0

    @property
    def centroid(self) -> Point2D:
        """Calculate polygon centroid."""
        return Point2D(x=float(np.mean(self.vertices[:, 0])), y=float(np.mean(self.vertices[:, 1])))

    @property
    def perimeter(self) -> float:
        """Calculate polygon perimeter."""
        n = len(self.vertices)
        if n < 2:
            return 0.0

        perim = 0.0
        for i in range(n):
            j = (i + 1) % n
            dx = self.vertices[j][0] - self.vertices[i][0]
            dy = self.vertices[j][1] - self.vertices[i][1]
            perim += np.sqrt(dx * dx + dy * dy)

        return perim

    def __repr__(self) -> str:
        return f"Polygon(vertices={self.num_vertices})"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box.

    Attributes:
        min_point: Minimum corner
        max_point: Maximum corner
    """

    min_point: Point2D
    max_point: Point2D

    @property
    def width(self) -> float:
        """Get box width."""
        return self.max_point.x - self.min_point.x

    @property
    def height(self) -> float:
        """Get box height."""
        return self.max_point.y - self.min_point.y

    @property
    def center(self) -> Point2D:
        """Get box center."""
        return Point2D(
            x=(self.min_point.x + self.max_point.x) / 2, y=(self.min_point.y + self.max_point.y) / 2
        )

    def __repr__(self) -> str:
        return f"BoundingBox({self.min_point} to {self.max_point})"


# ============================================================================
# LAYER 1: PRIMITIVE CONSTRUCTION
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(x: float, y: float) -> Point2D",
    deterministic=True,
    doc="Create a 2D point",
)
def point2d(x: float, y: float) -> Point2D:
    """Create a 2D point in Cartesian coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Point2D instance

    Example:
        p = point2d(x=3.0, y=4.0)
    """
    return Point2D(x=x, y=y)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(x: float, y: float, z: float) -> Point3D",
    deterministic=True,
    doc="Create a 3D point",
)
def point3d(x: float, y: float, z: float) -> Point3D:
    """Create a 3D point in Cartesian coordinates.

    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        Point3D instance

    Example:
        p = point3d(x=1.0, y=2.0, z=3.0)
    """
    return Point3D(x=x, y=y, z=z)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(start: Point2D, end: Point2D) -> Line2D",
    deterministic=True,
    doc="Create a 2D line segment",
)
def line2d(start: Point2D, end: Point2D) -> Line2D:
    """Create a 2D line segment from two points.

    Args:
        start: Start point
        end: End point

    Returns:
        Line2D instance

    Example:
        line = line2d(
            start=point2d(0.0, 0.0),
            end=point2d(1.0, 1.0)
        )
    """
    return Line2D(start=start, end=end)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point2D, radius: float) -> Circle",
    deterministic=True,
    doc="Create a circle",
)
def circle(center: Point2D, radius: float) -> Circle:
    """Create a circle from center point and radius.

    Args:
        center: Center point
        radius: Circle radius (must be positive)

    Returns:
        Circle instance

    Example:
        circ = circle(
            center=point2d(0.0, 0.0),
            radius=5.0
        )
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    return Circle(center=center, radius=radius)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point2D, width: float, height: float, rotation: float) -> Rectangle",
    deterministic=True,
    doc="Create a rectangle",
)
def rectangle(center: Point2D, width: float, height: float, rotation: float = 0.0) -> Rectangle:
    """Create a rectangle from center, dimensions, and rotation.

    Args:
        center: Center point
        width: Rectangle width (must be positive)
        height: Rectangle height (must be positive)
        rotation: Rotation angle in radians (default: 0)

    Returns:
        Rectangle instance

    Example:
        rect = rectangle(
            center=point2d(0.0, 0.0),
            width=10.0,
            height=5.0,
            rotation=np.pi / 4
        )
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")

    return Rectangle(center=center, width=width, height=height, rotation=rotation)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(vertices: ndarray) -> Polygon",
    deterministic=True,
    doc="Create a polygon from vertices",
)
def polygon(vertices: np.ndarray) -> Polygon:
    """Create a polygon from an array of vertices.

    Args:
        vertices: Array of vertices with shape [N, 2] where N >= 3

    Returns:
        Polygon instance

    Example:
        # Triangle
        tri = polygon(vertices=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ]))
    """
    vertices = np.asarray(vertices)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f"Vertices must have shape [N, 2], got {vertices.shape}")

    if len(vertices) < 3:
        raise ValueError(f"Polygon must have at least 3 vertices, got {len(vertices)}")

    return Polygon(vertices=vertices)


@operator(
    domain="geometry",
    category=OpCategory.CONSTRUCT,
    signature="(center: Point2D, radius: float, num_sides: int) -> Polygon",
    deterministic=True,
    doc="Create a regular polygon",
)
def regular_polygon(center: Point2D, radius: float, num_sides: int) -> Polygon:
    """Create a regular polygon (equal sides and angles).

    Args:
        center: Center point
        radius: Radius (distance from center to vertices)
        num_sides: Number of sides (must be >= 3)

    Returns:
        Polygon instance

    Example:
        # Regular hexagon
        hex = regular_polygon(
            center=point2d(0.0, 0.0),
            radius=1.0,
            num_sides=6
        )
    """
    if num_sides < 3:
        raise ValueError(f"Regular polygon must have at least 3 sides, got {num_sides}")

    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    # Generate vertices in counter-clockwise order
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertices = np.zeros((num_sides, 2))
    vertices[:, 0] = center.x + radius * np.cos(angles)
    vertices[:, 1] = center.y + radius * np.sin(angles)

    return Polygon(vertices=vertices)


# ============================================================================
# LAYER 2: TRANSFORMATIONS
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point2D, dx: float, dy: float) -> Point2D",
    deterministic=True,
    doc="Translate a 2D point",
)
def translate_point2d(point: Point2D, dx: float, dy: float) -> Point2D:
    """Translate a 2D point by offset.

    Args:
        point: Point to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated point

    Example:
        p2 = translate_point2d(p1, dx=5.0, dy=3.0)
    """
    return Point2D(x=point.x + dx, y=point.y + dy, frame=point.frame)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(circle: Circle, dx: float, dy: float) -> Circle",
    deterministic=True,
    doc="Translate a circle",
)
def translate_circle(circle: Circle, dx: float, dy: float) -> Circle:
    """Translate a circle by offset.

    Args:
        circle: Circle to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated circle

    Example:
        c2 = translate_circle(c1, dx=10.0, dy=0.0)
    """
    new_center = translate_point2d(circle.center, dx, dy)
    return Circle(center=new_center, radius=circle.radius)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(rect: Rectangle, dx: float, dy: float) -> Rectangle",
    deterministic=True,
    doc="Translate a rectangle",
)
def translate_rectangle(rect: Rectangle, dx: float, dy: float) -> Rectangle:
    """Translate a rectangle by offset.

    Args:
        rect: Rectangle to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated rectangle

    Example:
        r2 = translate_rectangle(r1, dx=5.0, dy=5.0)
    """
    new_center = translate_point2d(rect.center, dx, dy)
    return Rectangle(
        center=new_center, width=rect.width, height=rect.height, rotation=rect.rotation
    )


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(poly: Polygon, dx: float, dy: float) -> Polygon",
    deterministic=True,
    doc="Translate a polygon",
)
def translate_polygon(poly: Polygon, dx: float, dy: float) -> Polygon:
    """Translate a polygon by offset.

    Args:
        poly: Polygon to translate
        dx: X offset
        dy: Y offset

    Returns:
        Translated polygon

    Example:
        p2 = translate_polygon(p1, dx=2.0, dy=-3.0)
    """
    new_vertices = poly.vertices + np.array([dx, dy])
    return Polygon(vertices=new_vertices)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point2D, center: Point2D, angle: float) -> Point2D",
    deterministic=True,
    doc="Rotate a 2D point around a center",
)
def rotate_point2d(point: Point2D, center: Point2D, angle: float) -> Point2D:
    """Rotate a 2D point around a center point.

    Args:
        point: Point to rotate
        center: Center of rotation
        angle: Rotation angle in radians (counter-clockwise)

    Returns:
        Rotated point

    Example:
        # Rotate 90 degrees counter-clockwise
        p2 = rotate_point2d(p1, center=point2d(0, 0), angle=np.pi/2)
    """
    # Translate to origin
    dx = point.x - center.x
    dy = point.y - center.y

    # Rotate
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    new_x = dx * cos_a - dy * sin_a
    new_y = dx * sin_a + dy * cos_a

    # Translate back
    return Point2D(x=new_x + center.x, y=new_y + center.y, frame=point.frame)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(circle: Circle, center: Point2D, angle: float) -> Circle",
    deterministic=True,
    doc="Rotate a circle around a center",
)
def rotate_circle(circle: Circle, center: Point2D, angle: float) -> Circle:
    """Rotate a circle around a center point.

    Args:
        circle: Circle to rotate
        center: Center of rotation
        angle: Rotation angle in radians

    Returns:
        Rotated circle

    Note:
        Only the circle's center is rotated; radius remains unchanged.
    """
    new_center = rotate_point2d(circle.center, center, angle)
    return Circle(center=new_center, radius=circle.radius)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(rect: Rectangle, angle: float) -> Rectangle",
    deterministic=True,
    doc="Rotate a rectangle around its center",
)
def rotate_rectangle(rect: Rectangle, angle: float) -> Rectangle:
    """Rotate a rectangle around its center.

    Args:
        rect: Rectangle to rotate
        angle: Rotation angle in radians

    Returns:
        Rotated rectangle

    Example:
        # Rotate 45 degrees
        r2 = rotate_rectangle(r1, angle=np.pi/4)
    """
    return Rectangle(
        center=rect.center, width=rect.width, height=rect.height, rotation=rect.rotation + angle
    )


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(poly: Polygon, center: Point2D, angle: float) -> Polygon",
    deterministic=True,
    doc="Rotate a polygon around a center",
)
def rotate_polygon(poly: Polygon, center: Point2D, angle: float) -> Polygon:
    """Rotate a polygon around a center point.

    Args:
        poly: Polygon to rotate
        center: Center of rotation
        angle: Rotation angle in radians

    Returns:
        Rotated polygon

    Example:
        p2 = rotate_polygon(p1, center=p1.centroid, angle=np.pi/6)
    """
    # Translate to origin
    vertices = poly.vertices - np.array([center.x, center.y])

    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate
    rotated = vertices @ rot_matrix.T

    # Translate back
    new_vertices = rotated + np.array([center.x, center.y])

    return Polygon(vertices=new_vertices)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(circle: Circle, center: Point2D, scale: float) -> Circle",
    deterministic=True,
    doc="Scale a circle from a center point",
)
def scale_circle(circle: Circle, center: Point2D, scale: float) -> Circle:
    """Scale a circle from a center point.

    Args:
        circle: Circle to scale
        center: Center of scaling
        scale: Scale factor (must be positive)

    Returns:
        Scaled circle

    Example:
        # Double the size
        c2 = scale_circle(c1, center=c1.center, scale=2.0)
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")

    # Scale center position
    dx = (circle.center.x - center.x) * scale
    dy = (circle.center.y - center.y) * scale

    new_center = Point2D(x=center.x + dx, y=center.y + dy)
    new_radius = circle.radius * scale

    return Circle(center=new_center, radius=new_radius)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(rect: Rectangle, center: Point2D, scale_x: float, scale_y: float) -> Rectangle",
    deterministic=True,
    doc="Scale a rectangle from a center point",
)
def scale_rectangle(rect: Rectangle, center: Point2D, scale_x: float, scale_y: float) -> Rectangle:
    """Scale a rectangle from a center point.

    Args:
        rect: Rectangle to scale
        center: Center of scaling
        scale_x: X scale factor
        scale_y: Y scale factor

    Returns:
        Scaled rectangle

    Example:
        # Scale width by 2, height by 1.5
        r2 = scale_rectangle(r1, center=r1.center, scale_x=2.0, scale_y=1.5)
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Scale factors must be positive, got {scale_x}, {scale_y}")

    # Scale center position
    dx = (rect.center.x - center.x) * scale_x
    dy = (rect.center.y - center.y) * scale_y

    new_center = Point2D(x=center.x + dx, y=center.y + dy)

    return Rectangle(
        center=new_center,
        width=rect.width * scale_x,
        height=rect.height * scale_y,
        rotation=rect.rotation,
    )


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(poly: Polygon, center: Point2D, scale_x: float, scale_y: float) -> Polygon",
    deterministic=True,
    doc="Scale a polygon from a center point",
)
def scale_polygon(poly: Polygon, center: Point2D, scale_x: float, scale_y: float) -> Polygon:
    """Scale a polygon from a center point.

    Args:
        poly: Polygon to scale
        center: Center of scaling
        scale_x: X scale factor
        scale_y: Y scale factor

    Returns:
        Scaled polygon

    Example:
        p2 = scale_polygon(p1, center=p1.centroid, scale_x=2.0, scale_y=2.0)
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Scale factors must be positive, got {scale_x}, {scale_y}")

    # Translate to origin (convert to float to avoid casting issues)
    vertices = poly.vertices.astype(np.float64) - np.array([center.x, center.y])

    # Scale
    vertices[:, 0] *= scale_x
    vertices[:, 1] *= scale_y

    # Translate back
    new_vertices = vertices + np.array([center.x, center.y])

    return Polygon(vertices=new_vertices)


# ============================================================================
# LAYER 3: SPATIAL QUERIES
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(p1: Point2D, p2: Point2D) -> float",
    deterministic=True,
    doc="Calculate Euclidean distance between two 2D points",
)
def distance_point_point(p1: Point2D, p2: Point2D) -> float:
    """Calculate Euclidean distance between two 2D points.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Distance between points

    Example:
        dist = distance_point_point(
            point2d(0, 0),
            point2d(3, 4)
        )  # Returns 5.0
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return float(np.sqrt(dx * dx + dy * dy))


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, line: Line2D) -> float",
    deterministic=True,
    doc="Calculate distance from point to line segment",
)
def distance_point_line(point: Point2D, line: Line2D) -> float:
    """Calculate shortest distance from point to line segment.

    Args:
        point: Point to measure from
        line: Line segment

    Returns:
        Shortest distance

    Example:
        dist = distance_point_line(
            point=point2d(0, 1),
            line=line2d(point2d(-1, 0), point2d(1, 0))
        )  # Returns 1.0
    """
    # Vector from line start to point
    px = point.x - line.start.x
    py = point.y - line.start.y

    # Vector from line start to end
    lx = line.end.x - line.start.x
    ly = line.end.y - line.start.y

    # Line length squared
    line_len_sq = lx * lx + ly * ly

    if line_len_sq == 0:
        # Line is a point
        return distance_point_point(point, line.start)

    # Project point onto line (clamped to [0, 1])
    t = max(0, min(1, (px * lx + py * ly) / line_len_sq))

    # Closest point on line
    closest_x = line.start.x + t * lx
    closest_y = line.start.y + t * ly

    # Distance to closest point
    dx = point.x - closest_x
    dy = point.y - closest_y

    return float(np.sqrt(dx * dx + dy * dy))


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, circle: Circle) -> float",
    deterministic=True,
    doc="Calculate distance from point to circle perimeter",
)
def distance_point_circle(point: Point2D, circle: Circle) -> float:
    """Calculate distance from point to circle perimeter.

    Args:
        point: Point to measure from
        circle: Circle

    Returns:
        Distance to circle perimeter (negative if inside)

    Example:
        dist = distance_point_circle(
            point=point2d(5, 0),
            circle=circle(center=point2d(0, 0), radius=3)
        )  # Returns 2.0
    """
    dist_to_center = distance_point_point(point, circle.center)
    return dist_to_center - circle.radius


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(c1: Circle, c2: Circle) -> Optional[Tuple[Point2D, Point2D]]",
    deterministic=True,
    doc="Find intersection points between two circles",
)
def intersect_circle_circle(c1: Circle, c2: Circle) -> Optional[Tuple[Point2D, Point2D]]:
    """Find intersection points between two circles.

    Args:
        c1: First circle
        c2: Second circle

    Returns:
        Tuple of two intersection points, or None if no intersection

    Example:
        points = intersect_circle_circle(circle1, circle2)
        if points:
            p1, p2 = points
    """
    # Distance between centers
    d = distance_point_point(c1.center, c2.center)

    # Check if circles intersect
    if d > c1.radius + c2.radius or d < abs(c1.radius - c2.radius) or d == 0:
        return None

    # Find intersection points using analytical solution
    a = (c1.radius * c1.radius - c2.radius * c2.radius + d * d) / (2 * d)
    h = np.sqrt(c1.radius * c1.radius - a * a)

    # Point on line between centers
    dx = c2.center.x - c1.center.x
    dy = c2.center.y - c1.center.y

    px = c1.center.x + a * dx / d
    py = c1.center.y + a * dy / d

    # Perpendicular offset
    offset_x = h * dy / d
    offset_y = -h * dx / d

    p1 = Point2D(x=px + offset_x, y=py + offset_y)
    p2 = Point2D(x=px - offset_x, y=py - offset_y)

    return (p1, p2)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, circle: Circle) -> bool",
    deterministic=True,
    doc="Check if point is inside circle",
)
def contains_circle_point(circle: Circle, point: Point2D) -> bool:
    """Check if a circle contains a point.

    Args:
        circle: Circle
        point: Point to test

    Returns:
        True if point is inside or on circle boundary

    Example:
        inside = contains_circle_point(
            circle=circle(center=point2d(0, 0), radius=5),
            point=point2d(3, 4)
        )  # True (distance is 5.0)
    """
    dist = distance_point_point(circle.center, point)
    return dist <= circle.radius


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, poly: Polygon) -> bool",
    deterministic=True,
    doc="Check if point is inside polygon (ray casting)",
)
def contains_polygon_point(poly: Polygon, point: Point2D) -> bool:
    """Check if a polygon contains a point using ray casting algorithm.

    Args:
        poly: Polygon
        point: Point to test

    Returns:
        True if point is inside polygon

    Algorithm:
        Casts a ray from the point to infinity and counts edge crossings.
        Odd number of crossings = inside, even = outside.

    Example:
        inside = contains_polygon_point(triangle, point2d(0.5, 0.5))
    """
    n = len(poly.vertices)
    inside = False

    x, y = point.x, point.y

    j = n - 1
    for i in range(n):
        xi, yi = poly.vertices[i]
        xj, yj = poly.vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, rect: Rectangle) -> bool",
    deterministic=True,
    doc="Check if point is inside rectangle",
)
def contains_rectangle_point(rect: Rectangle, point: Point2D) -> bool:
    """Check if a rectangle contains a point.

    Args:
        rect: Rectangle
        point: Point to test

    Returns:
        True if point is inside rectangle

    Example:
        inside = contains_rectangle_point(rect, point2d(1, 1))
    """
    # If rectangle is not rotated, use simple bounds check
    if rect.rotation == 0:
        hw, hh = rect.width / 2, rect.height / 2
        dx = abs(point.x - rect.center.x)
        dy = abs(point.y - rect.center.y)
        return bool(dx <= hw and dy <= hh)

    # For rotated rectangle, transform point to local space
    # Translate to origin
    dx = point.x - rect.center.x
    dy = point.y - rect.center.y

    # Rotate by -rotation
    cos_r = np.cos(-rect.rotation)
    sin_r = np.sin(-rect.rotation)

    local_x = dx * cos_r - dy * sin_r
    local_y = dx * sin_r + dy * cos_r

    # Check bounds in local space
    hw, hh = rect.width / 2, rect.height / 2
    return bool(abs(local_x) <= hw and abs(local_y) <= hh)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, circle: Circle) -> Point2D",
    deterministic=True,
    doc="Find closest point on circle to given point",
)
def closest_point_circle(circle: Circle, point: Point2D) -> Point2D:
    """Find closest point on circle perimeter to a given point.

    Args:
        circle: Circle
        point: Point to find closest point to

    Returns:
        Closest point on circle

    Example:
        closest = closest_point_circle(
            circle=circle(center=point2d(0, 0), radius=5),
            point=point2d(10, 0)
        )  # Returns point2d(5, 0)
    """
    # Direction from center to point
    dx = point.x - circle.center.x
    dy = point.y - circle.center.y

    dist = np.sqrt(dx * dx + dy * dy)

    if dist == 0:
        # Point is at center, return any point on circle
        return Point2D(x=circle.center.x + circle.radius, y=circle.center.y)

    # Normalize and scale by radius
    scale = circle.radius / dist

    return Point2D(x=circle.center.x + dx * scale, y=circle.center.y + dy * scale)


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(point: Point2D, line: Line2D) -> Point2D",
    deterministic=True,
    doc="Find closest point on line segment to given point",
)
def closest_point_line(line: Line2D, point: Point2D) -> Point2D:
    """Find closest point on line segment to a given point.

    Args:
        line: Line segment
        point: Point to find closest point to

    Returns:
        Closest point on line

    Example:
        closest = closest_point_line(
            line=line2d(point2d(0, 0), point2d(10, 0)),
            point=point2d(5, 5)
        )  # Returns point2d(5, 0)
    """
    # Vector from line start to point
    px = point.x - line.start.x
    py = point.y - line.start.y

    # Vector from line start to end
    lx = line.end.x - line.start.x
    ly = line.end.y - line.start.y

    # Line length squared
    line_len_sq = lx * lx + ly * ly

    if line_len_sq == 0:
        # Line is a point
        return line.start

    # Project point onto line (clamped to [0, 1])
    t = max(0, min(1, (px * lx + py * ly) / line_len_sq))

    # Closest point on line
    return Point2D(x=line.start.x + t * lx, y=line.start.y + t * ly)


# ============================================================================
# LAYER 4: COORDINATE CONVERSIONS
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point2D) -> Tuple[float, float]",
    deterministic=True,
    doc="Convert Cartesian coordinates to polar (r, theta)",
)
def cartesian_to_polar(point: Point2D) -> Tuple[float, float]:
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        point: Point in Cartesian coordinates

    Returns:
        Tuple of (radius, angle) where angle is in radians

    Example:
        r, theta = cartesian_to_polar(point2d(3, 4))
        # r = 5.0, theta ≈ 0.927 radians (53.13 degrees)
    """
    r = np.sqrt(point.x * point.x + point.y * point.y)
    theta = np.arctan2(point.y, point.x)
    return (float(r), float(theta))


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(r: float, theta: float) -> Point2D",
    deterministic=True,
    doc="Convert polar coordinates to Cartesian (x, y)",
)
def polar_to_cartesian(r: float, theta: float) -> Point2D:
    """Convert polar coordinates to Cartesian coordinates.

    Args:
        r: Radius
        theta: Angle in radians

    Returns:
        Point in Cartesian coordinates

    Example:
        point = polar_to_cartesian(r=5.0, theta=np.pi/4)
        # Returns point2d(3.536, 3.536)
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return Point2D(x=float(x), y=float(y), frame=CoordinateFrame.POLAR)


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(point: Point3D) -> Tuple[float, float, float]",
    deterministic=True,
    doc="Convert Cartesian 3D to spherical (r, theta, phi)",
)
def cartesian_to_spherical(point: Point3D) -> Tuple[float, float, float]:
    """Convert Cartesian 3D coordinates to spherical coordinates.

    Args:
        point: Point in Cartesian coordinates

    Returns:
        Tuple of (r, theta, phi) where:
        - r: Radial distance
        - theta: Azimuthal angle (0 to 2π)
        - phi: Polar angle from z-axis (0 to π)

    Example:
        r, theta, phi = cartesian_to_spherical(point3d(1, 1, 1))
    """
    r = np.sqrt(point.x * point.x + point.y * point.y + point.z * point.z)
    theta = np.arctan2(point.y, point.x)
    phi = np.arccos(point.z / r) if r > 0 else 0.0

    return (float(r), float(theta), float(phi))


@operator(
    domain="geometry",
    category=OpCategory.TRANSFORM,
    signature="(r: float, theta: float, phi: float) -> Point3D",
    deterministic=True,
    doc="Convert spherical coordinates to Cartesian 3D",
)
def spherical_to_cartesian(r: float, theta: float, phi: float) -> Point3D:
    """Convert spherical coordinates to Cartesian 3D coordinates.

    Args:
        r: Radial distance
        theta: Azimuthal angle (radians)
        phi: Polar angle from z-axis (radians)

    Returns:
        Point in Cartesian coordinates

    Example:
        point = spherical_to_cartesian(r=1.0, theta=0, phi=np.pi/2)
        # Returns point on unit sphere
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return Point3D(x=float(x), y=float(y), z=float(z), frame=CoordinateFrame.SPHERICAL)


# ============================================================================
# LAYER 5: GEOMETRIC PROPERTIES
# ============================================================================


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon]) -> float",
    deterministic=True,
    doc="Calculate area of a 2D shape",
)
def area(shape: Union[Circle, Rectangle, Polygon]) -> float:
    """Calculate area of a 2D shape.

    Args:
        shape: Circle, Rectangle, or Polygon

    Returns:
        Area of the shape

    Example:
        a = area(circle(center=point2d(0, 0), radius=5))  # 78.54
    """
    return shape.area


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon]) -> float",
    deterministic=True,
    doc="Calculate perimeter of a 2D shape",
)
def perimeter(shape: Union[Circle, Rectangle, Polygon]) -> float:
    """Calculate perimeter of a 2D shape.

    Args:
        shape: Circle, Rectangle, or Polygon

    Returns:
        Perimeter (or circumference for circles)

    Example:
        p = perimeter(rectangle(point2d(0,0), width=4, height=3))  # 14
    """
    if isinstance(shape, Circle):
        return shape.circumference
    elif isinstance(shape, (Rectangle, Polygon)):
        return shape.perimeter
    else:
        raise TypeError(f"Cannot calculate perimeter for {type(shape)}")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Rectangle, Polygon]) -> Point2D",
    deterministic=True,
    doc="Calculate centroid of a shape",
)
def centroid(shape: Union[Rectangle, Polygon]) -> Point2D:
    """Calculate centroid (center of mass) of a shape.

    Args:
        shape: Rectangle or Polygon

    Returns:
        Centroid point

    Example:
        c = centroid(triangle)
    """
    if isinstance(shape, Rectangle):
        return shape.center
    elif isinstance(shape, Polygon):
        return shape.centroid
    else:
        raise TypeError(f"Cannot calculate centroid for {type(shape)}")


@operator(
    domain="geometry",
    category=OpCategory.QUERY,
    signature="(shape: Union[Circle, Rectangle, Polygon]) -> BoundingBox",
    deterministic=True,
    doc="Calculate axis-aligned bounding box of a shape",
)
def bounding_box(shape: Union[Circle, Rectangle, Polygon]) -> BoundingBox:
    """Calculate axis-aligned bounding box of a shape.

    Args:
        shape: Circle, Rectangle, or Polygon

    Returns:
        Bounding box

    Example:
        bbox = bounding_box(circle(center=point2d(0, 0), radius=5))
    """
    if isinstance(shape, Circle):
        return BoundingBox(
            min_point=Point2D(x=shape.center.x - shape.radius, y=shape.center.y - shape.radius),
            max_point=Point2D(x=shape.center.x + shape.radius, y=shape.center.y + shape.radius),
        )
    elif isinstance(shape, Rectangle):
        verts = shape.get_vertices()
        return BoundingBox(
            min_point=Point2D(x=float(np.min(verts[:, 0])), y=float(np.min(verts[:, 1]))),
            max_point=Point2D(x=float(np.max(verts[:, 0])), y=float(np.max(verts[:, 1]))),
        )
    elif isinstance(shape, Polygon):
        return BoundingBox(
            min_point=Point2D(
                x=float(np.min(shape.vertices[:, 0])), y=float(np.min(shape.vertices[:, 1]))
            ),
            max_point=Point2D(
                x=float(np.max(shape.vertices[:, 0])), y=float(np.max(shape.vertices[:, 1]))
            ),
        )
    else:
        raise TypeError(f"Cannot calculate bounding box for {type(shape)}")


# ============================================================================
# EXPORTS (for registry discovery)
# ============================================================================

# Primitives
__all__ = [
    # Types
    "Point2D",
    "Point3D",
    "Line2D",
    "Circle",
    "Rectangle",
    "Polygon",
    "BoundingBox",
    "CoordinateFrame",
    # Construction
    "point2d",
    "point3d",
    "line2d",
    "circle",
    "rectangle",
    "polygon",
    "regular_polygon",
    # Transformations
    "translate_point2d",
    "translate_circle",
    "translate_rectangle",
    "translate_polygon",
    "rotate_point2d",
    "rotate_circle",
    "rotate_rectangle",
    "rotate_polygon",
    "scale_circle",
    "scale_rectangle",
    "scale_polygon",
    # Spatial queries
    "distance_point_point",
    "distance_point_line",
    "distance_point_circle",
    "intersect_circle_circle",
    "contains_circle_point",
    "contains_polygon_point",
    "contains_rectangle_point",
    "closest_point_circle",
    "closest_point_line",
    # Coordinate conversions
    "cartesian_to_polar",
    "polar_to_cartesian",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    # Properties
    "area",
    "perimeter",
    "centroid",
    "bounding_box",
]
