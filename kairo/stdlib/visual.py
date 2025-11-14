"""Visual operations implementation.

This module provides visualization capabilities including field colorization,
PNG output, and interactive real-time display for the MVP.
"""

from typing import Optional, Union, Callable, Tuple
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

    # ========================================================================
    # VISUAL EXTENSIONS (v0.6.0)
    # ========================================================================

    @staticmethod
    def agents(agents, width: int = 512, height: int = 512,
               position_property: str = 'pos',
               color_property: Optional[str] = None,
               size_property: Optional[str] = None,
               color: tuple = (1.0, 1.0, 1.0),
               size: float = 2.0,
               palette: str = "viridis",
               background: tuple = (0.0, 0.0, 0.0),
               bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
               trail: bool = False,
               trail_length: int = 10,
               trail_alpha: float = 0.5) -> Visual:
        """Render agents as points or circles.

        Args:
            agents: Agents instance to visualize
            width: Output image width in pixels
            height: Output image height in pixels
            position_property: Name of position property (default: 'pos')
            color_property: Name of property to colorize by (optional)
            size_property: Name of property to size by (optional)
            color: Default color as (R, G, B) in [0, 1] (used if color_property=None)
            size: Default point size in pixels (used if size_property=None)
            palette: Color palette name for color_property mapping
            background: Background color as (R, G, B) in [0, 1]
            bounds: ((xmin, xmax), (ymin, ymax)) for position mapping, auto if None
            trail: If True, render agent trails (requires 'trail' property)
            trail_length: Number of trail points to render
            trail_alpha: Alpha transparency for trail

        Returns:
            Visual representation of agents

        Example:
            # Render agents with velocity-based coloring
            vis = visual.agents(
                agents,
                color_property='vel_mag',
                size=3.0,
                palette='viridis'
            )
        """
        from .agents import Agents

        if not isinstance(agents, Agents):
            raise TypeError(f"Expected Agents, got {type(agents)}")

        # Get positions
        positions = agents.get(position_property)
        if len(positions.shape) != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"Position property must be (N, 2) array, got shape {positions.shape}"
            )

        # Determine bounds
        if bounds is None:
            xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
            ymin, ymax = np.min(positions[:, 1]), np.max(positions[:, 1])

            # Add 10% padding
            x_padding = (xmax - xmin) * 0.1
            y_padding = (ymax - ymin) * 0.1
            bounds = ((xmin - x_padding, xmax + x_padding),
                     (ymin - y_padding, ymax + y_padding))
        else:
            (xmin, xmax), (ymin, ymax) = bounds

        # Create output image
        img = np.zeros((height, width, 3), dtype=np.float32)
        img[:, :, :] = background

        # Map positions to pixel coordinates
        x_norm = (positions[:, 0] - xmin) / (xmax - xmin)
        y_norm = (positions[:, 1] - ymin) / (ymax - ymin)

        px = (x_norm * (width - 1)).astype(int)
        py = ((1.0 - y_norm) * (height - 1)).astype(int)  # Flip Y axis

        # Clip to image bounds
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)

        # Determine colors
        if color_property is not None:
            # Color by property
            color_values = agents.get(color_property)

            # Handle vector properties (use magnitude)
            if len(color_values.shape) > 1:
                color_values = np.linalg.norm(color_values, axis=1)

            # Normalize to [0, 1]
            vmin, vmax = np.min(color_values), np.max(color_values)
            if vmax - vmin < 1e-10:
                color_norm = np.zeros_like(color_values)
            else:
                color_norm = (color_values - vmin) / (vmax - vmin)

            # Map to palette
            palette_colors = np.array(VisualOperations.PALETTES[palette])
            n_colors = len(palette_colors)

            indices = color_norm * (n_colors - 1)
            idx_low = np.floor(indices).astype(int)
            idx_high = np.minimum(idx_low + 1, n_colors - 1)
            frac = indices - idx_low

            colors = np.zeros((len(agents.get(position_property)), 3), dtype=np.float32)
            for c in range(3):
                colors[:, c] = (
                    palette_colors[idx_low, c] * (1 - frac) +
                    palette_colors[idx_high, c] * frac
                )
        else:
            # Use default color for all agents
            colors = np.tile(color, (len(positions), 1))

        # Determine sizes
        if size_property is not None:
            size_values = agents.get(size_property)

            # Handle vector properties (use magnitude)
            if len(size_values.shape) > 1:
                size_values = np.linalg.norm(size_values, axis=1)

            # Normalize and scale
            vmin, vmax = np.min(size_values), np.max(size_values)
            if vmax - vmin < 1e-10:
                sizes = np.ones(len(size_values)) * size
            else:
                size_norm = (size_values - vmin) / (vmax - vmin)
                sizes = size_norm * size * 2  # Scale up to 2x base size
        else:
            sizes = np.ones(len(positions)) * size

        # Render agents as circles
        for i in range(len(positions)):
            agent_size = int(sizes[i])
            agent_color = colors[i]

            # Draw filled circle
            y, x = py[i], px[i]
            for dy in range(-agent_size, agent_size + 1):
                for dx in range(-agent_size, agent_size + 1):
                    if dx*dx + dy*dy <= agent_size * agent_size:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            img[ny, nx] = agent_color

        return Visual(img)

    @staticmethod
    def layer(visual: Optional[Visual] = None, width: int = 512, height: int = 512,
              background: tuple = (0.0, 0.0, 0.0)) -> Visual:
        """Create a visual layer for composition.

        Args:
            visual: Existing visual to convert to layer (optional)
            width: Layer width if creating new layer
            height: Layer height if creating new layer
            background: Background color as (R, G, B) in [0, 1]

        Returns:
            Visual layer

        Example:
            # Create empty layer
            layer1 = visual.layer(width=512, height=512)

            # Convert existing visual to layer
            layer2 = visual.layer(existing_visual)
        """
        if visual is not None:
            if not isinstance(visual, Visual):
                raise TypeError(f"Expected Visual, got {type(visual)}")
            return visual.copy()
        else:
            # Create new empty layer
            img = np.zeros((height, width, 3), dtype=np.float32)
            img[:, :, :] = background
            return Visual(img)

    @staticmethod
    def composite(*layers: Visual, mode: str = "over",
                  opacity: Optional[Union[float, list]] = None) -> Visual:
        """Composite multiple visual layers.

        Args:
            *layers: Visual layers to composite (bottom to top)
            mode: Blending mode ("over", "add", "multiply", "screen", "overlay")
            opacity: Opacity for each layer (0.0 to 1.0), or single value for all

        Returns:
            Composited visual

        Example:
            # Composite field and agents
            field_vis = visual.colorize(temperature, palette="fire")
            agent_vis = visual.agents(particles, color=(1, 1, 1))
            result = visual.composite(field_vis, agent_vis, mode="add")
        """
        if len(layers) == 0:
            raise ValueError("At least one layer required")

        # Validate all layers are Visual instances
        for i, layer in enumerate(layers):
            if not isinstance(layer, Visual):
                raise TypeError(f"Layer {i} is not a Visual instance")

        # Check all layers have same dimensions
        base_shape = layers[0].shape
        for i, layer in enumerate(layers[1:], 1):
            if layer.shape != base_shape:
                raise ValueError(
                    f"Layer {i} has shape {layer.shape}, expected {base_shape}"
                )

        # Handle opacity
        if opacity is None:
            opacities = [1.0] * len(layers)
        elif isinstance(opacity, (int, float)):
            opacities = [float(opacity)] * len(layers)
        else:
            if len(opacity) != len(layers):
                raise ValueError(
                    f"opacity list length {len(opacity)} doesn't match layers {len(layers)}"
                )
            opacities = list(opacity)

        # Start with first layer
        result = layers[0].data.copy() * opacities[0]

        # Composite remaining layers
        for i, layer in enumerate(layers[1:], 1):
            alpha = opacities[i]
            top = layer.data
            bottom = result

            if mode == "over":
                # Standard alpha compositing (over operator)
                result = bottom * (1 - alpha) + top * alpha
            elif mode == "add":
                # Additive blending
                result = bottom + top * alpha
            elif mode == "multiply":
                # Multiply blending
                result = bottom * (1 - alpha + top * alpha)
            elif mode == "screen":
                # Screen blending
                result = 1 - (1 - bottom) * (1 - top * alpha)
            elif mode == "overlay":
                # Overlay blending
                mask = bottom < 0.5
                result = np.where(
                    mask,
                    2 * bottom * top * alpha + bottom * (1 - alpha),
                    1 - 2 * (1 - bottom) * (1 - top) * alpha + bottom * (1 - alpha)
                )
            else:
                raise ValueError(
                    f"Unknown blending mode: {mode}. "
                    f"Supported: 'over', 'add', 'multiply', 'screen', 'overlay'"
                )

        return Visual(result)

    @staticmethod
    def video(frames: Union[list, Callable[[], Optional[Visual]]],
              path: str,
              fps: int = 30,
              format: str = "auto",
              max_frames: Optional[int] = None) -> None:
        """Export animation sequence to video file.

        Supports MP4 and GIF output formats.

        Args:
            frames: List of Visual frames or generator function
            path: Output file path
            fps: Frames per second
            format: Output format ("auto", "mp4", "gif") - auto infers from extension
            max_frames: Maximum number of frames to export (for generators)

        Raises:
            ImportError: If imageio is not installed

        Example:
            # From list of frames
            frames = [generate_frame(i) for i in range(100)]
            visual.video(frames, "output.mp4", fps=30)

            # From generator
            def gen_frames():
                temp = field.random((128, 128))
                for i in range(100):
                    temp = field.diffuse(temp, rate=0.1)
                    yield visual.colorize(temp, palette="fire")

            visual.video(gen_frames, "output.gif", fps=10)
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. "
                "Install with: pip install imageio imageio-ffmpeg"
            )

        # Infer format from path if auto
        if format == "auto":
            if path.endswith(".mp4"):
                format = "mp4"
            elif path.endswith(".gif"):
                format = "gif"
            else:
                format = "mp4"  # Default

        format = format.lower()

        # Collect frames
        if callable(frames):
            # Generator function
            frame_list = []
            count = 0
            while True:
                if max_frames is not None and count >= max_frames:
                    break

                frame = frames()
                if frame is None:
                    break

                if not isinstance(frame, Visual):
                    raise TypeError(f"Frame {count} is not a Visual instance")

                frame_list.append(frame)
                count += 1
        else:
            # List of frames
            frame_list = list(frames)

            # Validate all frames
            for i, frame in enumerate(frame_list):
                if not isinstance(frame, Visual):
                    raise TypeError(f"Frame {i} is not a Visual instance")

        if len(frame_list) == 0:
            raise ValueError("No frames to export")

        print(f"Exporting {len(frame_list)} frames to {path}...")

        # Convert frames to 8-bit RGB
        rgb_frames = []
        for frame in frame_list:
            # Apply gamma correction
            srgb = VisualOperations._linear_to_srgb(frame.data)

            # Convert to 8-bit
            rgb_8bit = (srgb * 255).astype(np.uint8)
            rgb_frames.append(rgb_8bit)

        # Write video
        if format == "mp4":
            # MP4 requires ffmpeg
            try:
                imageio.mimwrite(
                    path,
                    rgb_frames,
                    fps=fps,
                    codec='libx264',
                    quality=8,
                    pixelformat='yuv420p'
                )
            except Exception as e:
                # Fall back to basic MP4 if codec not available
                imageio.mimwrite(path, rgb_frames, fps=fps)
        elif format == "gif":
            # GIF export
            imageio.mimwrite(
                path,
                rgb_frames,
                fps=fps,
                loop=0  # Infinite loop
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: 'mp4', 'gif'")

        print(f"Video export complete: {path}")


# Create singleton instance for use as 'visual' namespace
visual = VisualOperations()
