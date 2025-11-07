"""Visual operations implementation.

This module provides visualization capabilities including field colorization,
PNG output, and interactive real-time display for the MVP.
"""

from typing import Optional, Union, Callable
import numpy as np
import time


class Visual:
    """Opaque visual representation (linear RGB).

    Stores rendered image data ready for output.
    """

    def __init__(self, data: np.ndarray):
        """Initialize visual.

        Args:
            data: RGB image data (shape: (height, width, 3), dtype: float32, range: [0, 1])
        """
        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(f"Visual data must be (height, width, 3), got {data.shape}")

        self.data = np.clip(data, 0.0, 1.0).astype(np.float32)
        self.shape = data.shape[:2]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get image width."""
        return self.shape[1]

    def copy(self) -> 'Visual':
        """Create a copy of this visual."""
        return Visual(self.data.copy())

    def __repr__(self) -> str:
        return f"Visual(shape={self.shape})"


class VisualOperations:
    """Namespace for visual operations (accessed as 'visual' in DSL)."""

    # Color palettes (linear RGB values)
    PALETTES = {
        "grayscale": [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ],
        "fire": [
            (0.0, 0.0, 0.0),      # Black
            (0.5, 0.0, 0.0),      # Dark red
            (1.0, 0.0, 0.0),      # Red
            (1.0, 0.5, 0.0),      # Orange
            (1.0, 1.0, 0.0),      # Yellow
            (1.0, 1.0, 1.0),      # White
        ],
        "viridis": [
            (0.267, 0.005, 0.329),  # Dark purple
            (0.283, 0.141, 0.458),  # Purple
            (0.254, 0.266, 0.530),  # Blue-purple
            (0.207, 0.372, 0.554),  # Blue
            (0.164, 0.471, 0.558),  # Cyan-blue
            (0.135, 0.568, 0.551),  # Cyan
            (0.196, 0.664, 0.523),  # Green-cyan
            (0.395, 0.762, 0.420),  # Green
            (0.671, 0.867, 0.253),  # Yellow-green
            (0.993, 0.906, 0.144),  # Yellow
        ],
        "coolwarm": [
            (0.23, 0.30, 0.75),     # Cool blue
            (0.57, 0.77, 0.87),     # Light blue
            (0.87, 0.87, 0.87),     # White
            (0.96, 0.68, 0.52),     # Light orange
            (0.71, 0.02, 0.15),     # Warm red
        ],
    }

    @staticmethod
    def colorize(field, palette: str = "grayscale",
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None) -> Visual:
        """Map scalar field to colors using a palette.

        Args:
            field: Field2D to colorize
            palette: Palette name ("grayscale", "fire", "viridis", "coolwarm")
            vmin: Minimum value for mapping (default: field min)
            vmax: Maximum value for mapping (default: field max)

        Returns:
            Visual representation of the field
        """
        from .field import Field2D

        if not isinstance(field, Field2D):
            raise TypeError(f"Expected Field2D, got {type(field)}")

        # Get field data
        data = field.data

        # Handle multi-channel fields (use magnitude)
        if len(data.shape) == 3:
            data = np.linalg.norm(data, axis=2)

        # Normalize to [0, 1]
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        # Avoid division by zero
        if vmax - vmin < 1e-10:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - vmin) / (vmax - vmin)

        normalized = np.clip(normalized, 0.0, 1.0)

        # Get palette colors
        if palette not in VisualOperations.PALETTES:
            raise ValueError(f"Unknown palette: {palette}. Available: {list(VisualOperations.PALETTES.keys())}")

        palette_colors = np.array(VisualOperations.PALETTES[palette])
        n_colors = len(palette_colors)

        # Map normalized values to palette indices
        indices = normalized * (n_colors - 1)
        idx_low = np.floor(indices).astype(int)
        idx_high = np.minimum(idx_low + 1, n_colors - 1)
        frac = indices - idx_low

        # Interpolate between palette colors
        h, w = normalized.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for c in range(3):
            rgb[:, :, c] = (
                palette_colors[idx_low, c] * (1 - frac) +
                palette_colors[idx_high, c] * frac
            )

        return Visual(rgb)

    @staticmethod
    def output(visual: Visual, path: str, format: str = "auto") -> None:
        """Save visual to file.

        Args:
            visual: Visual to save
            path: Output file path
            format: Output format ("auto", "png", "jpg") - auto infers from extension

        Raises:
            ImportError: If PIL/Pillow is not installed
        """
        if not isinstance(visual, Visual):
            raise TypeError(f"Expected Visual, got {type(visual)}")

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for visual output. "
                "Install with: pip install Pillow"
            )

        # Infer format from path if auto
        if format == "auto":
            if path.endswith(".png"):
                format = "png"
            elif path.endswith(".jpg") or path.endswith(".jpeg"):
                format = "jpeg"
            else:
                format = "png"  # Default

        # Normalize format for PIL
        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG"
        }
        pil_format = format_map.get(format.lower(), "PNG")

        # Convert linear RGB to sRGB (gamma correction)
        srgb = VisualOperations._linear_to_srgb(visual.data)

        # Convert to 8-bit
        rgb_8bit = (srgb * 255).astype(np.uint8)

        # Save image (Pillow auto-detects RGB from uint8 array shape)
        img = Image.fromarray(rgb_8bit)
        img.save(path, pil_format)

        print(f"Saved visual to: {path}")

    @staticmethod
    def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB with gamma correction.

        Args:
            linear: Linear RGB values in [0, 1]

        Returns:
            sRGB values in [0, 1]
        """
        # sRGB gamma correction
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055
        )
        return np.clip(srgb, 0.0, 1.0)

    @staticmethod
    def display(frame_generator: Callable[[], Optional[Visual]],
                title: str = "Creative Computation DSL",
                target_fps: int = 30,
                scale: int = 2) -> None:
        """Display simulation in real-time interactive window.

        Args:
            frame_generator: Callable that generates frames. Should return Visual or None to quit.
            title: Window title
            target_fps: Target frames per second
            scale: Scale factor for display (1 = native resolution)

        Controls:
            SPACE: Pause/Resume
            RIGHT: Step forward (when paused)
            UP/DOWN: Increase/decrease speed
            Q/ESC: Quit

        Example:
            >>> def generate_frames():
            ...     temp = field.random((128, 128), seed=42)
            ...     while True:
            ...         temp = field.diffuse(temp, rate=0.1, dt=0.1)
            ...         yield visual.colorize(temp, palette="fire")
            >>>
            >>> gen = generate_frames()
            >>> visual.display(lambda: next(gen))
        """
        # Input validation
        if not callable(frame_generator):
            raise TypeError(f"frame_generator must be callable, got {type(frame_generator)}")

        if not isinstance(title, str):
            raise TypeError(f"title must be str, got {type(title)}")

        if not isinstance(target_fps, int) or target_fps <= 0:
            raise ValueError(f"target_fps must be positive integer, got {target_fps}")

        if not isinstance(scale, int) or scale <= 0:
            raise ValueError(f"scale must be positive integer, got {scale}")

        try:
            import pygame
        except ImportError:
            raise ImportError(
                "pygame is required for interactive display. "
                "Install with: pip install pygame"
            )

        # Initialize pygame
        pygame.init()

        # Get first frame to determine size
        first_frame = frame_generator()
        if first_frame is None:
            return

        if not isinstance(first_frame, Visual):
            raise TypeError(f"frame_generator must return Visual, got {type(first_frame)}")

        # Create display window
        width, height = first_frame.width * scale, first_frame.height * scale
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()

        # Create font for UI
        font = pygame.font.Font(None, 24)

        # State
        paused = False
        current_fps = target_fps
        fps_frame_count = 0  # For FPS calculation (resets every second)
        total_frames = 0  # Total frames generated
        fps_timer = time.time()
        actual_fps = 0.0
        current_visual = first_frame

        try:
            running = True
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_RIGHT and paused:
                            # Step forward one frame
                            new_frame = frame_generator()
                            if new_frame is not None:
                                current_visual = new_frame
                                total_frames += 1
                                fps_frame_count += 1
                        elif event.key == pygame.K_UP:
                            current_fps = min(current_fps + 5, 120)
                        elif event.key == pygame.K_DOWN:
                            current_fps = max(current_fps - 5, 1)
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                # Generate next frame (if not paused)
                if not paused:
                    new_frame = frame_generator()
                    if new_frame is None:
                        running = False
                        continue
                    current_visual = new_frame
                    total_frames += 1
                    fps_frame_count += 1

                # Convert visual to pygame surface
                srgb = VisualOperations._linear_to_srgb(current_visual.data)
                rgb_8bit = (srgb * 255).astype(np.uint8)

                # Create surface and scale
                surf = pygame.surfarray.make_surface(np.transpose(rgb_8bit, (1, 0, 2)))
                if scale != 1:
                    surf = pygame.transform.scale(surf, (width, height))

                # Draw to screen
                screen.blit(surf, (0, 0))

                # Draw UI overlay
                now = time.time()
                if now - fps_timer >= 1.0:
                    actual_fps = fps_frame_count / (now - fps_timer)
                    fps_frame_count = 0
                    fps_timer = now

                # Status text
                status_lines = [
                    f"FPS: {actual_fps:.1f} / {current_fps}",
                    f"Frame: {total_frames}" if paused else "",
                    "PAUSED" if paused else "RUNNING",
                    "",
                    "Controls:",
                    "SPACE: Pause/Resume",
                    "→: Step (paused)",
                    "↑↓: Speed",
                    "Q: Quit"
                ]

                y_offset = 10
                for line in status_lines:
                    if line:
                        # Draw with black background for readability
                        text = font.render(line, True, (255, 255, 255))
                        text_bg = pygame.Surface((text.get_width() + 10, text.get_height() + 4))
                        text_bg.set_alpha(180)
                        text_bg.fill((0, 0, 0))
                        screen.blit(text_bg, (5, y_offset))
                        screen.blit(text, (10, y_offset + 2))
                    y_offset += 22

                pygame.display.flip()
                clock.tick(current_fps)

        finally:
            pygame.quit()


# Create singleton instance for use as 'visual' namespace
visual = VisualOperations()
