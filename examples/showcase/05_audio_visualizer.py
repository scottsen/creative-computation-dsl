"""Audio Visualizer - Advanced Cross-Domain Showcase

This example demonstrates the power of combining multiple Kairo domains:
- Audio synthesis and processing
- FFT spectral analysis
- Field operations for audio-reactive effects
- Cellular automata driven by audio
- Palette and color for stunning visuals
- Image composition and export

Creates beautiful visualizations that react to audio:
- Spectrum analyzers with smooth animations
- Audio-reactive cellular automata
- Waveform visualizations
- Beat-synchronized patterns
- Multi-domain integration showcase
"""

import numpy as np
from kairo.stdlib import audio, field, cellular, palette, color, image, noise
from kairo.stdlib.field import Field2D


def compute_fft_spectrum(audio_buffer, window_size=2048, hop_size=512):
    """Compute FFT spectrum from audio buffer.

    Args:
        audio_buffer: Audio buffer to analyze
        window_size: FFT window size
        hop_size: Hop size for STFT

    Returns:
        2D array of spectral magnitudes (time x frequency)
    """
    data = audio_buffer.data
    num_frames = (len(data) - window_size) // hop_size + 1

    spectrum = np.zeros((num_frames, window_size // 2))

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size

        if end > len(data):
            break

        # Extract window
        window = data[start:end]

        # Apply Hann window
        window = window * np.hanning(window_size)

        # Compute FFT
        fft = np.fft.rfft(window)
        magnitude = np.abs(fft)[:window_size // 2]

        # Convert to dB scale
        magnitude = 20 * np.log10(magnitude + 1e-10)

        spectrum[i, :] = magnitude

    return spectrum


def create_spectrum_analyzer(audio_buffer, width=800, height=400,
                             colormap='plasma'):
    """Create a classic spectrum analyzer visualization.

    Args:
        audio_buffer: Audio to visualize
        width: Output width
        height: Output height
        colormap: Color palette

    Returns:
        RGB image of spectrum analyzer
    """
    # Compute spectrum
    spectrum = compute_fft_spectrum(audio_buffer)

    # Resize to target dimensions
    # Interpolate spectrum to match output size
    time_frames, freq_bins = spectrum.shape

    # Create visualization field
    if time_frames < width:
        # Upsample in time
        indices = np.linspace(0, time_frames - 1, width).astype(int)
        spectrum_resized = spectrum[indices, :]
    else:
        # Downsample in time
        indices = np.linspace(0, time_frames - 1, width).astype(int)
        spectrum_resized = spectrum[indices, :]

    if freq_bins < height:
        # Upsample in frequency
        spectrum_final = np.zeros((width, height))
        for i in range(width):
            spectrum_final[i, :] = np.interp(
                np.linspace(0, freq_bins - 1, height),
                np.arange(freq_bins),
                spectrum_resized[i, :]
            )
    else:
        # Downsample in frequency
        spectrum_final = spectrum_resized[:, :height]

    # Transpose to get (frequency, time) orientation
    spectrum_final = spectrum_final.T

    # Normalize
    spectrum_final = spectrum_final - spectrum_final.min()
    if spectrum_final.max() > 0:
        spectrum_final = spectrum_final / spectrum_final.max()

    # Apply colormap
    pal = palette.create_gradient(colormap, 256)
    img = palette.apply(pal, spectrum_final)

    return img


def create_waveform_visualization(audio_buffer, width=800, height=200,
                                  colormap='viridis'):
    """Create waveform visualization.

    Args:
        audio_buffer: Audio to visualize
        width: Output width
        height: Output height
        colormap: Color palette

    Returns:
        RGB image of waveform
    """
    data = audio_buffer.data
    samples_per_pixel = len(data) // width

    # Create field
    waveform_field = np.zeros((height, width), dtype=np.float32)

    for i in range(width):
        start = i * samples_per_pixel
        end = start + samples_per_pixel

        if end > len(data):
            break

        # Get min/max for this window
        window = data[start:end]
        min_val = np.min(window)
        max_val = np.max(window)

        # Map to pixel coordinates
        min_y = int((min_val + 1.0) * 0.5 * height)
        max_y = int((max_val + 1.0) * 0.5 * height)

        # Clamp
        min_y = np.clip(min_y, 0, height - 1)
        max_y = np.clip(max_y, 0, height - 1)

        # Draw vertical line
        waveform_field[min_y:max_y+1, i] = 1.0

    # Apply colormap
    pal = palette.create_gradient(colormap, 256)
    img = palette.apply(pal, waveform_field)

    return img


def audio_reactive_cellular_automata(audio_buffer, ca_size=200, duration=5.0):
    """Create cellular automaton that reacts to audio.

    Audio amplitude controls birth rate / density.

    Args:
        audio_buffer: Audio signal
        ca_size: CA grid size
        duration: Duration in seconds

    Returns:
        List of CA frames
    """
    sample_rate = audio_buffer.sample_rate
    data = audio_buffer.data

    # Number of CA steps
    fps = 30
    num_frames = int(duration * fps)
    samples_per_frame = len(data) // num_frames

    # Initialize CA
    field_ca, rule = cellular.game_of_life((ca_size, ca_size),
                                           density=0.3, seed=42)

    frames = []

    for frame_idx in range(num_frames):
        # Get audio segment for this frame
        start = frame_idx * samples_per_frame
        end = start + samples_per_frame

        if end > len(data):
            break

        audio_segment = data[start:end]
        amplitude = np.mean(np.abs(audio_segment))

        # Evolve CA
        field_ca = cellular.step(field_ca, rule)

        # Add random cells based on audio amplitude
        if amplitude > 0.1:
            num_cells = int(amplitude * 100)
            rng = np.random.RandomState(frame_idx)
            for _ in range(num_cells):
                x = rng.randint(0, ca_size)
                y = rng.randint(0, ca_size)
                field_ca.data[y, x] = 1

        frames.append(field_ca.copy())

    return frames


def create_beat_synchronized_patterns(audio_buffer, pattern_size=300):
    """Create patterns synchronized to audio beats.

    Uses energy in signal to trigger pattern changes.

    Args:
        audio_buffer: Audio signal
        pattern_size: Size of pattern grid

    Returns:
        RGB image of final pattern
    """
    data = audio_buffer.data

    # Compute energy envelope
    window_size = 2048
    hop_size = 512
    num_windows = (len(data) - window_size) // hop_size

    energy = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = data[start:end]
        energy[i] = np.sum(window ** 2)

    # Normalize energy
    energy = energy / (energy.max() + 1e-10)

    # Create field that accumulates based on energy
    pattern = Field2D(np.zeros((pattern_size, pattern_size), dtype=np.float32))

    for i, e in enumerate(energy):
        if e > 0.5:  # Beat detected
            # Add radial pattern
            cx, cy = pattern_size // 2, pattern_size // 2
            radius = int(e * 50)

            y, x = np.ogrid[:pattern_size, :pattern_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

            pattern.data[mask] += e * 0.5

    # Normalize
    pattern.data = np.clip(pattern.data, 0, 1)

    # Apply colormap
    pal = palette.create_gradient('hot', 256)
    img = palette.apply(pal, pattern.data)

    return img


def create_audio_reactive_field(audio_buffer, width=400, height=400):
    """Create diffusion field driven by audio spectrum.

    Audio frequencies create heat sources that diffuse.

    Args:
        audio_buffer: Audio signal
        width: Field width
        height: Field height

    Returns:
        RGB image of final field state
    """
    # Compute spectrum
    spectrum = compute_fft_spectrum(audio_buffer, window_size=2048)

    # Create field
    heat_field = field.alloc((height, width), dtype=np.float32, fill_value=0.0)

    # Add heat based on spectrum
    num_freq_bins = spectrum.shape[1]

    for time_idx in range(min(spectrum.shape[0], 100)):  # Limit iterations
        # Get spectrum at this time
        spectrum_slice = spectrum[time_idx, :]

        # Normalize
        spectrum_slice = spectrum_slice - spectrum_slice.min()
        if spectrum_slice.max() > 0:
            spectrum_slice = spectrum_slice / spectrum_slice.max()

        # Add heat at positions corresponding to frequencies
        for freq_idx, magnitude in enumerate(spectrum_slice[:50]):  # Use lower freqs
            if magnitude > 0.3:
                # Position based on frequency
                x = int((freq_idx / 50) * width)
                y = height // 2

                # Add heat
                heat_field.data[y-5:y+5, x-5:x+5] += magnitude * 0.5

        # Diffuse
        heat_field = field.diffuse(heat_field, diffusion_coeff=0.1, dt=0.1)

        # Decay
        heat_field.data *= 0.98

    # Normalize
    heat_field.data = np.clip(heat_field.data, 0, 1)

    # Apply colormap
    pal = palette.create_gradient('inferno', 256)
    img = palette.apply(pal, heat_field.data)

    return img


def demo_spectrum_analyzer():
    """Demo: Classic spectrum analyzer."""
    print("Creating spectrum analyzer...")

    # Generate test audio: multiple sine waves
    duration = 3.0
    sample_rate = 44100

    # Create chord (C major)
    c_note = audio.sine(freq=261.63, duration=duration, sample_rate=sample_rate)
    e_note = audio.sine(freq=329.63, duration=duration, sample_rate=sample_rate)
    g_note = audio.sine(freq=392.00, duration=duration, sample_rate=sample_rate)

    # Mix
    chord = audio.AudioBuffer(
        data=(c_note.data + e_note.data + g_note.data) / 3.0,
        sample_rate=sample_rate
    )

    # Add some noise for texture
    noise_data = np.random.randn(len(chord.data)) * 0.05
    chord.data += noise_data

    # Create visualization
    img = create_spectrum_analyzer(chord, width=800, height=400,
                                   colormap='plasma')

    # Save
    image.save(img, "output_audio_spectrum_analyzer.png")
    print("   ✓ Saved: output_audio_spectrum_analyzer.png")


def demo_waveform():
    """Demo: Waveform visualization."""
    print("Creating waveform visualization...")

    # Generate test audio: amplitude modulated sine
    duration = 2.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Carrier frequency
    carrier = np.sin(2 * np.pi * 440 * t)

    # Modulation
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)

    # AM synthesis
    am_signal = carrier * modulator

    buf = audio.AudioBuffer(data=am_signal, sample_rate=sample_rate)

    # Visualize
    img = create_waveform_visualization(buf, width=800, height=200,
                                       colormap='cool')

    image.save(img, "output_audio_waveform.png")
    print("   ✓ Saved: output_audio_waveform.png")


def demo_audio_reactive_ca():
    """Demo: Audio-reactive cellular automaton."""
    print("Creating audio-reactive cellular automaton...")

    # Generate rhythmic audio
    duration = 5.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Create rhythm: kick drum pattern
    kick_pattern = np.zeros_like(t)
    beat_interval = 0.5  # 120 BPM
    for beat_time in np.arange(0, duration, beat_interval):
        beat_idx = int(beat_time * sample_rate)
        # Exponential decay envelope
        decay_samples = int(0.2 * sample_rate)
        envelope = np.exp(-10 * np.arange(decay_samples) / decay_samples)
        # Sine wave at low frequency
        kick = np.sin(2 * np.pi * 60 * np.arange(decay_samples) / sample_rate) * envelope

        end_idx = min(beat_idx + decay_samples, len(kick_pattern))
        kick_pattern[beat_idx:end_idx] += kick[:end_idx - beat_idx]

    buf = audio.AudioBuffer(data=kick_pattern, sample_rate=sample_rate)

    # Create CA frames
    ca_frames = audio_reactive_cellular_automata(buf, ca_size=200, duration=duration)

    # Visualize a few key frames
    frame_indices = [0, len(ca_frames)//4, len(ca_frames)//2, 3*len(ca_frames)//4, -1]

    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx < len(ca_frames):
            ca_field = ca_frames[frame_idx]

            # Visualize
            pal = palette.create_gradient('magma', 256)
            img = palette.apply(pal, ca_field.data.astype(np.float32))

            output_path = f"output_audio_reactive_ca_frame{idx:02d}.png"
            image.save(img, output_path)
            print(f"   ✓ Saved: {output_path}")


def demo_beat_patterns():
    """Demo: Beat-synchronized patterns."""
    print("Creating beat-synchronized patterns...")

    # Generate beat pattern
    duration = 4.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Create drum pattern
    pattern = np.zeros_like(t)

    # Kick on 1 and 3
    for beat in [0.0, 1.0, 2.0, 3.0]:
        idx = int(beat * sample_rate)
        decay = np.exp(-20 * np.arange(2000) / sample_rate)
        kick = np.sin(2 * np.pi * 80 * np.arange(2000) / sample_rate) * decay
        pattern[idx:idx+2000] += kick

    # Snare on 2 and 4
    for beat in [0.5, 1.5, 2.5, 3.5]:
        idx = int(beat * sample_rate)
        snare = np.random.randn(4000) * np.exp(-15 * np.arange(4000) / sample_rate)
        pattern[idx:idx+4000] += snare * 0.5

    buf = audio.AudioBuffer(data=pattern, sample_rate=sample_rate)

    # Create visualization
    img = create_beat_synchronized_patterns(buf, pattern_size=400)

    image.save(img, "output_audio_beat_patterns.png")
    print("   ✓ Saved: output_audio_beat_patterns.png")


def demo_audio_field_diffusion():
    """Demo: Audio-driven field diffusion."""
    print("Creating audio-driven field diffusion...")

    # Generate sweeping tone
    duration = 3.0
    sample_rate = 44100
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Frequency sweep
    freq_start = 100
    freq_end = 2000
    freq = freq_start + (freq_end - freq_start) * (t / duration)

    # Instantaneous phase
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate

    sweep = np.sin(phase)

    buf = audio.AudioBuffer(data=sweep, sample_rate=sample_rate)

    # Create field visualization
    img = create_audio_reactive_field(buf, width=400, height=400)

    image.save(img, "output_audio_field_diffusion.png")
    print("   ✓ Saved: output_audio_field_diffusion.png")


def main():
    """Run all audio visualizer demonstrations."""
    print("=" * 60)
    print("AUDIO VISUALIZER - CROSS-DOMAIN SHOWCASE")
    print("=" * 60)
    print()
    print("Domains: Audio + Field + Cellular + Palette + Image")
    print()

    # Demo 1: Spectrum analyzer
    print("Demo 1: Spectrum Analyzer")
    print("-" * 60)
    demo_spectrum_analyzer()
    print()

    # Demo 2: Waveform
    print("Demo 2: Waveform Visualization")
    print("-" * 60)
    demo_waveform()
    print()

    # Demo 3: Audio-reactive CA
    print("Demo 3: Audio-Reactive Cellular Automaton")
    print("-" * 60)
    demo_audio_reactive_ca()
    print()

    # Demo 4: Beat patterns
    print("Demo 4: Beat-Synchronized Patterns")
    print("-" * 60)
    demo_beat_patterns()
    print()

    # Demo 5: Field diffusion
    print("Demo 5: Audio-Driven Field Diffusion")
    print("-" * 60)
    demo_audio_field_diffusion()
    print()

    print("=" * 60)
    print("ALL AUDIO VISUALIZER DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("This showcase demonstrates:")
    print("  • Audio synthesis and analysis (FFT, spectrum)")
    print("  • Field operations (diffusion, heat propagation)")
    print("  • Cellular automata (audio-reactive patterns)")
    print("  • Color mapping and palettes")
    print("  • Cross-domain integration")
    print()
    print("Key insight: Temporal domains (audio) can drive")
    print("spatial domains (field, cellular) to create")
    print("stunning audio-reactive visualizations!")


if __name__ == "__main__":
    main()
