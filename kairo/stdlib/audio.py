"""Audio operations implementation using NumPy backend.

This module provides NumPy-based implementations of all core audio operations
for deterministic audio synthesis, including oscillators, filters, envelopes,
effects, and physical modeling primitives.

All operations follow the audio-rate model (44.1kHz default) with deterministic
semantics ensuring same seed = same output.
"""

from typing import Callable, Optional, Dict, Any, Tuple, Union
import numpy as np


# Default audio parameters
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_CONTROL_RATE = 1000  # Hz


class AudioBuffer:
    """Audio buffer representing a stream of samples.

    Represents audio-rate (Sig) or control-rate (Ctl) signals as NumPy arrays
    with associated sample rate and metadata.

    Example:
        # Create a 1-second buffer at 44.1kHz
        buf = AudioBuffer(
            data=np.zeros(44100),
            sample_rate=44100
        )
    """

    def __init__(self, data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """Initialize audio buffer.

        Args:
            data: NumPy array of samples (1D for mono, 2D for multi-channel)
            sample_rate: Sample rate in Hz
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        return len(self.data)

    @property
    def is_stereo(self) -> bool:
        """Check if buffer is stereo."""
        return len(self.data.shape) > 1 and self.data.shape[1] == 2

    def copy(self) -> 'AudioBuffer':
        """Create a deep copy of this buffer."""
        return AudioBuffer(data=self.data.copy(), sample_rate=self.sample_rate)

    def __repr__(self) -> str:
        """String representation."""
        channels = "stereo" if self.is_stereo else "mono"
        return f"AudioBuffer({channels}, {self.num_samples} samples, {self.sample_rate}Hz)"


class AudioOperations:
    """Namespace for audio operations (accessed as 'audio' in DSL)."""

    # ========================================================================
    # OSCILLATORS (Section 5.1)
    # ========================================================================

    @staticmethod
    def sine(freq: float = 440.0, phase: float = 0.0, duration: float = 1.0,
             sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate sine wave oscillator.

        Args:
            freq: Frequency in Hz
            phase: Initial phase in radians (0 to 2π)
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with sine wave

        Example:
            # A440 tone for 1 second
            tone = audio.sine(freq=440.0, duration=1.0)
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        data = np.sin(2.0 * np.pi * freq * t + phase)
        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
    def saw(freq: float = 440.0, duration: float = 1.0, blep: bool = True,
            sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate sawtooth wave oscillator.

        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            blep: Enable band-limiting (PolyBLEP)
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with sawtooth wave
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        if blep:
            # PolyBLEP sawtooth (band-limited)
            phase = (freq * t) % 1.0
            data = 2.0 * phase - 1.0

            # Simple PolyBLEP residual
            dt = freq / sample_rate
            for i in range(num_samples):
                t_norm = phase[i]
                if t_norm < dt:
                    t_norm = t_norm / dt
                    data[i] += t_norm + t_norm - t_norm * t_norm - 1.0
                elif t_norm > 1.0 - dt:
                    t_norm = (t_norm - 1.0) / dt
                    data[i] += t_norm * t_norm + t_norm + t_norm + 1.0
        else:
            # Naive sawtooth (aliased)
            phase = (freq * t) % 1.0
            data = 2.0 * phase - 1.0

        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
    def square(freq: float = 440.0, pwm: float = 0.5, duration: float = 1.0,
               sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate square wave oscillator.

        Args:
            freq: Frequency in Hz
            pwm: Pulse width modulation (0.0 to 1.0, 0.5 = 50% duty cycle)
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with square wave
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        phase = (freq * t) % 1.0

        data = np.where(phase < pwm, 1.0, -1.0)
        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
    def triangle(freq: float = 440.0, duration: float = 1.0,
                 sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate triangle wave oscillator.

        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with triangle wave
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        phase = (freq * t) % 1.0

        # Triangle: ramp up from -1 to 1, then down from 1 to -1
        data = np.where(phase < 0.5,
                       4.0 * phase - 1.0,  # Rising edge
                       3.0 - 4.0 * phase)   # Falling edge
        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
    def noise(noise_type: str = "white", seed: int = 0, duration: float = 1.0,
              sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate noise oscillator.

        Args:
            noise_type: Type of noise ("white", "pink", "brown")
            seed: Random seed for deterministic output
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with noise

        Example:
            # White noise, deterministic
            noise = audio.noise(noise_type="white", seed=42, duration=1.0)
        """
        rng = np.random.RandomState(seed)
        num_samples = int(duration * sample_rate)

        if noise_type == "white":
            data = rng.randn(num_samples)
        elif noise_type == "pink":
            # Simple pink noise approximation (1/f)
            white = rng.randn(num_samples)
            # Apply moving average filter for 1/f characteristic
            b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
            # Simple IIR filter implementation
            data = AudioOperations._apply_iir_filter(white, b, a)
        elif noise_type == "brown":
            # Brownian noise (integrated white noise)
            white = rng.randn(num_samples)
            data = np.cumsum(white)
            # Normalize to prevent drift
            data = data / (np.max(np.abs(data)) + 1e-6)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Normalize to [-1, 1]
        data = data / (np.max(np.abs(data)) + 1e-6)
        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
    def impulse(rate: float = 1.0, duration: float = 1.0,
                sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate impulse train.

        Args:
            rate: Impulse rate in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with impulse train
        """
        num_samples = int(duration * sample_rate)
        data = np.zeros(num_samples)

        # Place impulses at regular intervals
        interval = int(sample_rate / rate)
        if interval > 0:
            data[::interval] = 1.0

        return AudioBuffer(data=data, sample_rate=sample_rate)

    # ========================================================================
    # FILTERS (Section 5.2)
    # ========================================================================

    @staticmethod
    def lowpass(signal: AudioBuffer, cutoff: float = 2000.0, q: float = 0.707) -> AudioBuffer:
        """Apply lowpass filter.

        Args:
            signal: Input audio buffer
            cutoff: Cutoff frequency in Hz
            q: Quality factor (resonance)

        Returns:
            Filtered audio buffer

        Example:
            # Remove high frequencies above 2kHz
            filtered = audio.lowpass(signal, cutoff=2000.0)
        """
        # Biquad lowpass filter
        b, a = AudioOperations._biquad_lowpass(cutoff, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
    def highpass(signal: AudioBuffer, cutoff: float = 120.0, q: float = 0.707) -> AudioBuffer:
        """Apply highpass filter.

        Args:
            signal: Input audio buffer
            cutoff: Cutoff frequency in Hz
            q: Quality factor (resonance)

        Returns:
            Filtered audio buffer
        """
        # Biquad highpass filter
        b, a = AudioOperations._biquad_highpass(cutoff, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
    def bandpass(signal: AudioBuffer, center: float = 1000.0, q: float = 1.0) -> AudioBuffer:
        """Apply bandpass filter.

        Args:
            signal: Input audio buffer
            center: Center frequency in Hz
            q: Quality factor (bandwidth)

        Returns:
            Filtered audio buffer
        """
        # Biquad bandpass filter
        b, a = AudioOperations._biquad_bandpass(center, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
    def notch(signal: AudioBuffer, center: float = 1000.0, q: float = 1.0) -> AudioBuffer:
        """Apply notch (band-stop) filter.

        Args:
            signal: Input audio buffer
            center: Center frequency in Hz
            q: Quality factor (bandwidth)

        Returns:
            Filtered audio buffer
        """
        # Biquad notch filter
        b, a = AudioOperations._biquad_notch(center, q, signal.sample_rate)
        filtered = AudioOperations._apply_iir_filter(signal.data, b, a)
        return AudioBuffer(data=filtered, sample_rate=signal.sample_rate)

    @staticmethod
    def eq3(signal: AudioBuffer, bass: float = 0.0, mid: float = 0.0,
            treble: float = 0.0) -> AudioBuffer:
        """Apply 3-band equalizer.

        Args:
            signal: Input audio buffer
            bass: Bass gain in dB (-12 to +12)
            mid: Mid gain in dB (-12 to +12)
            treble: Treble gain in dB (-12 to +12)

        Returns:
            Equalized audio buffer
        """
        # Apply low shelf for bass
        if abs(bass) > 0.01:
            b, a = AudioOperations._biquad_low_shelf(100.0, bass, signal.sample_rate)
            signal = AudioBuffer(
                data=AudioOperations._apply_iir_filter(signal.data, b, a),
                sample_rate=signal.sample_rate
            )

        # Apply peaking filter for mids
        if abs(mid) > 0.01:
            b, a = AudioOperations._biquad_peaking(1000.0, mid, 1.0, signal.sample_rate)
            signal = AudioBuffer(
                data=AudioOperations._apply_iir_filter(signal.data, b, a),
                sample_rate=signal.sample_rate
            )

        # Apply high shelf for treble
        if abs(treble) > 0.01:
            b, a = AudioOperations._biquad_high_shelf(8000.0, treble, signal.sample_rate)
            signal = AudioBuffer(
                data=AudioOperations._apply_iir_filter(signal.data, b, a),
                sample_rate=signal.sample_rate
            )

        return signal

    # ========================================================================
    # ENVELOPES (Section 5.3)
    # ========================================================================

    @staticmethod
    def adsr(attack: float = 0.005, decay: float = 0.08, sustain: float = 0.7,
             release: float = 0.2, duration: float = 1.0,
             sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate ADSR envelope.

        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
            duration: Total duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with ADSR envelope

        Example:
            # Classic synth envelope
            env = audio.adsr(attack=0.01, decay=0.1, sustain=0.6, release=0.3, duration=1.0)
        """
        num_samples = int(duration * sample_rate)
        envelope = np.zeros(num_samples)

        # Calculate sample counts for each stage
        attack_samples = int(attack * sample_rate)
        decay_samples = int(decay * sample_rate)
        release_samples = int(release * sample_rate)

        # Ensure we don't overflow
        attack_samples = min(attack_samples, num_samples)
        decay_samples = min(decay_samples, num_samples - attack_samples)
        release_samples = min(release_samples, num_samples)

        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        sustain_samples = max(0, sustain_samples)

        idx = 0

        # Attack: 0 -> 1
        if attack_samples > 0:
            envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples

        # Decay: 1 -> sustain
        if decay_samples > 0:
            envelope[idx:idx+decay_samples] = np.linspace(1, sustain, decay_samples)
            idx += decay_samples

        # Sustain: hold at sustain level
        if sustain_samples > 0:
            envelope[idx:idx+sustain_samples] = sustain
            idx += sustain_samples

        # Release: sustain -> 0
        if release_samples > 0 and idx < num_samples:
            actual_release = min(release_samples, num_samples - idx)
            envelope[idx:idx+actual_release] = np.linspace(sustain, 0, actual_release)

        return AudioBuffer(data=envelope, sample_rate=sample_rate)

    @staticmethod
    def ar(attack: float = 0.005, release: float = 0.3, duration: float = 1.0,
           sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate AR (Attack-Release) envelope.

        Args:
            attack: Attack time in seconds
            release: Release time in seconds
            duration: Total duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with AR envelope
        """
        num_samples = int(duration * sample_rate)
        envelope = np.zeros(num_samples)

        attack_samples = int(attack * sample_rate)
        release_samples = int(release * sample_rate)

        attack_samples = min(attack_samples, num_samples)
        release_samples = min(release_samples, num_samples - attack_samples)

        idx = 0

        # Attack: 0 -> 1
        if attack_samples > 0:
            envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples

        # Release: 1 -> 0
        if release_samples > 0 and idx < num_samples:
            actual_release = min(release_samples, num_samples - idx)
            envelope[idx:idx+actual_release] = np.linspace(1, 0, actual_release)

        return AudioBuffer(data=envelope, sample_rate=sample_rate)

    @staticmethod
    def envexp(time_constant: float = 0.05, duration: float = 1.0,
               sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioBuffer:
        """Generate exponential decay envelope.

        Args:
            time_constant: Time constant (63% decay time) in seconds
            duration: Total duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            AudioBuffer with exponential envelope
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        envelope = np.exp(-t / time_constant)
        return AudioBuffer(data=envelope, sample_rate=sample_rate)

    # ========================================================================
    # EFFECTS (Section 5.4)
    # ========================================================================

    @staticmethod
    def delay(signal: AudioBuffer, time: float = 0.3, feedback: float = 0.3,
              mix: float = 0.25) -> AudioBuffer:
        """Apply delay effect.

        Args:
            signal: Input audio buffer
            time: Delay time in seconds
            feedback: Feedback amount (0.0 to <1.0)
            mix: Dry/wet mix (0.0 = dry, 1.0 = wet)

        Returns:
            Audio buffer with delay

        Example:
            # Classic slapback delay
            delayed = audio.delay(signal, time=0.125, feedback=0.3, mix=0.3)
        """
        delay_samples = int(time * signal.sample_rate)

        if delay_samples <= 0:
            return signal.copy()

        # Create delay line
        wet = np.zeros_like(signal.data)

        for i in range(len(signal.data)):
            wet[i] = signal.data[i]
            if i >= delay_samples:
                wet[i] += feedback * wet[i - delay_samples]

        # Mix dry and wet
        output = (1.0 - mix) * signal.data + mix * wet
        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    @staticmethod
    def reverb(signal: AudioBuffer, mix: float = 0.12, size: float = 0.8) -> AudioBuffer:
        """Apply reverb effect (Schroeder reverberator).

        Args:
            signal: Input audio buffer
            mix: Dry/wet mix (0.0 to 1.0)
            size: Room size (0.0 to 1.0)

        Returns:
            Audio buffer with reverb
        """
        # Simple Schroeder reverb with 4 comb filters and 2 allpass
        sr = signal.sample_rate

        # Comb filter delays (scaled by room size)
        comb_delays = [int(size * d) for d in [1557, 1617, 1491, 1422]]
        comb_gains = [0.805, 0.827, 0.783, 0.764]

        # Allpass delays
        allpass_delays = [int(size * d) for d in [225, 556]]
        allpass_gains = [0.7, 0.7]

        # Process through comb filters
        wet = np.zeros_like(signal.data)
        for delay, gain in zip(comb_delays, comb_gains):
            comb_out = np.zeros_like(signal.data)
            for i in range(len(signal.data)):
                comb_out[i] = signal.data[i]
                if i >= delay:
                    comb_out[i] += gain * comb_out[i - delay]
            wet += comb_out

        wet = wet / len(comb_delays)

        # Process through allpass filters
        for delay, gain in zip(allpass_delays, allpass_gains):
            allpass_out = np.zeros_like(wet)
            for i in range(len(wet)):
                if i >= delay:
                    allpass_out[i] = -gain * wet[i] + wet[i - delay] + gain * allpass_out[i - delay]
                else:
                    allpass_out[i] = -gain * wet[i]
            wet = allpass_out

        # Mix dry and wet
        output = (1.0 - mix) * signal.data + mix * wet
        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    @staticmethod
    def chorus(signal: AudioBuffer, rate: float = 0.3, depth: float = 0.008,
               mix: float = 0.25) -> AudioBuffer:
        """Apply chorus effect.

        Args:
            signal: Input audio buffer
            rate: LFO rate in Hz
            depth: Modulation depth in seconds
            mix: Dry/wet mix

        Returns:
            Audio buffer with chorus
        """
        # Generate LFO
        lfo = AudioOperations.sine(freq=rate, duration=signal.duration,
                                   sample_rate=signal.sample_rate)

        # Modulated delay line
        base_delay = 0.02  # 20ms base delay
        depth_samples = depth * signal.sample_rate
        base_samples = int(base_delay * signal.sample_rate)

        wet = np.zeros_like(signal.data)
        for i in range(len(signal.data)):
            # Calculate modulated delay
            mod_delay = int(base_samples + depth_samples * lfo.data[i])
            mod_delay = max(0, min(mod_delay, i))

            if i >= mod_delay:
                wet[i] = signal.data[i - mod_delay]

        # Mix dry and wet
        output = (1.0 - mix) * signal.data + mix * wet
        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    @staticmethod
    def flanger(signal: AudioBuffer, rate: float = 0.2, depth: float = 0.003,
                feedback: float = 0.25, mix: float = 0.5) -> AudioBuffer:
        """Apply flanger effect.

        Args:
            signal: Input audio buffer
            rate: LFO rate in Hz
            depth: Modulation depth in seconds
            feedback: Feedback amount
            mix: Dry/wet mix

        Returns:
            Audio buffer with flanger
        """
        # Similar to chorus but with shorter delay and feedback
        lfo = AudioOperations.sine(freq=rate, duration=signal.duration,
                                   sample_rate=signal.sample_rate)

        base_delay = 0.001  # 1ms base delay
        depth_samples = depth * signal.sample_rate
        base_samples = int(base_delay * signal.sample_rate)

        wet = np.zeros_like(signal.data)
        for i in range(len(signal.data)):
            mod_delay = int(base_samples + depth_samples * lfo.data[i])
            mod_delay = max(0, min(mod_delay, i))

            if i >= mod_delay:
                wet[i] = signal.data[i - mod_delay]
                if i >= mod_delay and mod_delay > 0:
                    wet[i] += feedback * wet[i - mod_delay]

        output = (1.0 - mix) * signal.data + mix * wet
        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    @staticmethod
    def drive(signal: AudioBuffer, amount: float = 0.5, shape: str = "tanh") -> AudioBuffer:
        """Apply distortion/drive.

        Args:
            signal: Input audio buffer
            amount: Drive amount (0.0 to 1.0)
            shape: Distortion shape ("tanh", "hard", "soft")

        Returns:
            Distorted audio buffer
        """
        gain = 1.0 + amount * 10.0
        driven = signal.data * gain

        if shape == "tanh":
            # Smooth saturation
            output = np.tanh(driven)
        elif shape == "hard":
            # Hard clipping
            output = np.clip(driven, -1.0, 1.0)
        elif shape == "soft":
            # Soft clipping (cubic)
            output = np.where(np.abs(driven) < 1.0,
                            driven - (driven ** 3) / 3.0,
                            np.sign(driven))
        else:
            raise ValueError(f"Unknown distortion shape: {shape}")

        # Compensate for gain
        output = output / (1.0 + amount)

        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    @staticmethod
    def limiter(signal: AudioBuffer, threshold: float = -1.0,
                release: float = 0.05) -> AudioBuffer:
        """Apply limiter/compressor.

        Args:
            signal: Input audio buffer
            threshold: Threshold in dB
            release: Release time in seconds

        Returns:
            Limited audio buffer
        """
        threshold_lin = AudioOperations.db2lin(threshold)

        # Simple peak limiter
        output = signal.data.copy()
        gain = 1.0
        release_coef = np.exp(-1.0 / (release * signal.sample_rate))

        for i in range(len(output)):
            # Detect peak
            peak = abs(output[i])

            # Calculate required gain reduction
            if peak > threshold_lin:
                target_gain = threshold_lin / (peak + 1e-6)
            else:
                target_gain = 1.0

            # Smooth gain changes
            if target_gain < gain:
                gain = target_gain  # Fast attack
            else:
                gain = target_gain + (gain - target_gain) * release_coef  # Slow release

            output[i] *= gain

        return AudioBuffer(data=output, sample_rate=signal.sample_rate)

    # ========================================================================
    # UTILITIES (Section 5.5)
    # ========================================================================

    @staticmethod
    def mix(*signals: AudioBuffer) -> AudioBuffer:
        """Mix multiple audio signals with gain compensation.

        Args:
            *signals: Audio buffers to mix

        Returns:
            Mixed audio buffer

        Example:
            # Mix three signals
            mixed = audio.mix(bass, lead, pad)
        """
        if not signals:
            raise ValueError("At least one signal required")

        # Ensure all signals have same length and sample rate
        sample_rate = signals[0].sample_rate
        max_len = max(s.num_samples for s in signals)

        # Sum with gain compensation
        output = np.zeros(max_len)
        for signal in signals:
            # Pad if needed
            if signal.num_samples < max_len:
                padded = np.pad(signal.data, (0, max_len - signal.num_samples))
                output += padded
            else:
                output += signal.data[:max_len]

        # Gain compensate by sqrt(N) to prevent clipping
        output = output / np.sqrt(len(signals))

        return AudioBuffer(data=output, sample_rate=sample_rate)

    @staticmethod
    def gain(signal: AudioBuffer, amount_db: float) -> AudioBuffer:
        """Apply gain in dB.

        Args:
            signal: Input audio buffer
            amount_db: Gain in decibels

        Returns:
            Audio buffer with gain applied
        """
        gain_lin = AudioOperations.db2lin(amount_db)
        return AudioBuffer(data=signal.data * gain_lin, sample_rate=signal.sample_rate)

    @staticmethod
    def pan(signal: AudioBuffer, position: float = 0.0) -> AudioBuffer:
        """Pan mono signal to stereo.

        Args:
            signal: Input audio buffer (mono)
            position: Pan position (-1.0 = left, 0.0 = center, 1.0 = right)

        Returns:
            Stereo audio buffer
        """
        # Constant power panning
        position = np.clip(position, -1.0, 1.0)
        angle = (position + 1.0) * np.pi / 4.0  # -1..1 -> 0..π/2

        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        stereo = np.zeros((signal.num_samples, 2))
        stereo[:, 0] = signal.data * left_gain
        stereo[:, 1] = signal.data * right_gain

        return AudioBuffer(data=stereo, sample_rate=signal.sample_rate)

    @staticmethod
    def clip(signal: AudioBuffer, limit: float = 0.98) -> AudioBuffer:
        """Hard clip signal.

        Args:
            signal: Input audio buffer
            limit: Clipping threshold (0.0 to 1.0)

        Returns:
            Clipped audio buffer
        """
        clipped = np.clip(signal.data, -limit, limit)
        return AudioBuffer(data=clipped, sample_rate=signal.sample_rate)

    @staticmethod
    def normalize(signal: AudioBuffer, target: float = 0.98) -> AudioBuffer:
        """Normalize signal to target peak level.

        Args:
            signal: Input audio buffer
            target: Target peak level (0.0 to 1.0)

        Returns:
            Normalized audio buffer
        """
        peak = np.max(np.abs(signal.data))
        if peak > 1e-6:
            gain = target / peak
            return AudioBuffer(data=signal.data * gain, sample_rate=signal.sample_rate)
        return signal.copy()

    @staticmethod
    def db2lin(db: float) -> float:
        """Convert decibels to linear gain.

        Args:
            db: Value in decibels

        Returns:
            Linear gain value
        """
        return 10.0 ** (db / 20.0)

    @staticmethod
    def lin2db(linear: float) -> float:
        """Convert linear gain to decibels.

        Args:
            linear: Linear gain value

        Returns:
            Value in decibels
        """
        return 20.0 * np.log10(linear + 1e-10)

    # ========================================================================
    # PHYSICAL MODELING (Section 7)
    # ========================================================================

    @staticmethod
    def string(excitation: AudioBuffer, freq: float, t60: float = 1.5,
               damping: float = 0.3) -> AudioBuffer:
        """Karplus-Strong string physical model.

        Args:
            excitation: Excitation signal (noise burst, pluck, etc.)
            freq: Fundamental frequency in Hz
            t60: Decay time (time to -60dB) in seconds
            damping: High-frequency damping (0.0 to 1.0)

        Returns:
            String resonance output

        Example:
            # Plucked string
            exc = audio.noise(seed=1, duration=0.01)
            exc = audio.lowpass(exc, cutoff=6000.0)
            string_sound = audio.string(exc, freq=220.0, t60=1.5)
        """
        delay_samples = int(excitation.sample_rate / freq)

        if delay_samples <= 0:
            return excitation.copy()

        # Karplus-Strong algorithm
        output = np.zeros(excitation.num_samples)
        delay_line = np.zeros(delay_samples)

        # Calculate feedback gain for desired T60
        feedback = 0.996 ** (1.0 / (t60 * freq))

        for i in range(excitation.num_samples):
            # Read from delay line
            delayed = delay_line[0]

            # Add excitation
            output[i] = excitation.data[i] + delayed

            # Lowpass filter for damping (averaging filter)
            if damping > 0:
                filtered = (delayed + delay_line[-1]) * 0.5
                filtered = delayed * (1.0 - damping) + filtered * damping
            else:
                filtered = delayed

            # Write to delay line
            delay_line = np.roll(delay_line, -1)
            delay_line[-1] = filtered * feedback

        return AudioBuffer(data=output, sample_rate=excitation.sample_rate)

    @staticmethod
    def modal(excitation: AudioBuffer, frequencies: list, decays: list,
              amplitudes: Optional[list] = None) -> AudioBuffer:
        """Modal synthesis (resonant body).

        Args:
            excitation: Excitation signal
            frequencies: List of modal frequencies in Hz
            decays: List of decay times in seconds for each mode
            amplitudes: Optional list of relative amplitudes (default: all 1.0)

        Returns:
            Modal synthesis output

        Example:
            # Bell-like sound
            exc = audio.impulse(rate=1.0, duration=0.001)
            bell = audio.modal(exc,
                              frequencies=[200, 370, 550, 720],
                              decays=[2.0, 1.5, 1.0, 0.8])
        """
        if amplitudes is None:
            amplitudes = [1.0] * len(frequencies)

        if len(frequencies) != len(decays) or len(frequencies) != len(amplitudes):
            raise ValueError("frequencies, decays, and amplitudes must have same length")

        output = np.zeros(excitation.num_samples)

        # Each mode is a decaying sinusoid
        for freq, decay, amp in zip(frequencies, decays, amplitudes):
            # Exponential decay envelope
            env = AudioOperations.envexp(time_constant=decay / 5.0,
                                        duration=excitation.duration,
                                        sample_rate=excitation.sample_rate)

            # Sinusoidal oscillator
            osc = AudioOperations.sine(freq=freq, duration=excitation.duration,
                                      sample_rate=excitation.sample_rate)

            # Apply envelope and amplitude
            mode_output = osc.data * env.data * amp

            # Convolve with excitation (simplified - just multiply)
            output += mode_output * np.mean(excitation.data)

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak

        return AudioBuffer(data=output, sample_rate=excitation.sample_rate)

    # ========================================================================
    # HELPER FUNCTIONS (Internal)
    # ========================================================================

    @staticmethod
    def _apply_iir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Apply IIR filter using Direct Form II."""
        # Normalize coefficients
        a0 = a[0]
        if abs(a0) < 1e-10:
            return signal.copy()

        b = b / a0
        a = a / a0

        # Initialize state
        n_b = len(b)
        n_a = len(a)
        max_order = max(n_b, n_a) - 1

        # Pad coefficients
        if n_b < max_order + 1:
            b = np.pad(b, (0, max_order + 1 - n_b))
        if n_a < max_order + 1:
            a = np.pad(a, (0, max_order + 1 - n_a))

        # Apply filter
        output = np.zeros_like(signal)
        state = np.zeros(max_order)

        for i in range(len(signal)):
            # Direct Form II
            w = signal[i]
            for j in range(1, len(a)):
                w -= a[j] * state[j - 1] if j - 1 < len(state) else 0

            output[i] = b[0] * w
            for j in range(1, len(b)):
                output[i] += b[j] * state[j - 1] if j - 1 < len(state) else 0

            # Update state
            state = np.roll(state, 1)
            state[0] = w

        return output

    @staticmethod
    def _biquad_lowpass(cutoff: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad lowpass filter coefficients."""
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = (1.0 - np.cos(w0)) / 2.0
        b1 = 1.0 - np.cos(w0)
        b2 = (1.0 - np.cos(w0)) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    @staticmethod
    def _biquad_highpass(cutoff: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad highpass filter coefficients."""
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = (1.0 + np.cos(w0)) / 2.0
        b1 = -(1.0 + np.cos(w0))
        b2 = (1.0 + np.cos(w0)) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    @staticmethod
    def _biquad_bandpass(center: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad bandpass filter coefficients."""
        w0 = 2.0 * np.pi * center / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    @staticmethod
    def _biquad_notch(center: float, q: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad notch filter coefficients."""
        w0 = 2.0 * np.pi * center / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = 1.0
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    @staticmethod
    def _biquad_low_shelf(cutoff: float, gain_db: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad low shelf filter coefficients."""
        A = np.sqrt(10.0 ** (gain_db / 20.0))
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / 2.0

        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    @staticmethod
    def _biquad_high_shelf(cutoff: float, gain_db: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad high shelf filter coefficients."""
        A = np.sqrt(10.0 ** (gain_db / 20.0))
        w0 = 2.0 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / 2.0

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    @staticmethod
    def _biquad_peaking(center: float, gain_db: float, q: float,
                       sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate biquad peaking filter coefficients."""
        A = np.sqrt(10.0 ** (gain_db / 20.0))
        w0 = 2.0 * np.pi * center / sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

    # ========================================================================
    # AUDIO I/O (v0.6.0)
    # ========================================================================

    @staticmethod
    def play(buffer: AudioBuffer, blocking: bool = True) -> None:
        """Play audio buffer in real-time.

        Args:
            buffer: Audio buffer to play
            blocking: If True, wait for playback to complete (default: True)

        Raises:
            ImportError: If sounddevice is not installed

        Example:
            # Generate and play a tone
            tone = audio.sine(freq=440.0, duration=1.0)
            audio.play(tone)
        """
        if not isinstance(buffer, AudioBuffer):
            raise TypeError(f"Expected AudioBuffer, got {type(buffer)}")

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install with: pip install sounddevice"
            )

        # Prepare data for playback
        # sounddevice expects shape (samples, channels) for stereo
        if buffer.is_stereo:
            data = buffer.data  # Already (samples, 2)
        else:
            data = buffer.data.reshape(-1, 1)  # Make (samples, 1) for mono

        # Play audio
        sd.play(data, samplerate=buffer.sample_rate, blocking=blocking)

    @staticmethod
    def save(buffer: AudioBuffer, path: str, format: str = "auto") -> None:
        """Save audio buffer to file.

        Supports WAV and FLAC formats with automatic format detection from file extension.

        Args:
            buffer: Audio buffer to save
            path: Output file path
            format: Output format ("auto", "wav", "flac") - auto infers from extension

        Raises:
            ImportError: If soundfile is not installed (for FLAC) or scipy (for WAV fallback)
            ValueError: If format is unsupported

        Example:
            # Generate and save audio
            tone = audio.sine(freq=440.0, duration=1.0)
            audio.save(tone, "output.wav")
            audio.save(tone, "output.flac")
        """
        if not isinstance(buffer, AudioBuffer):
            raise TypeError(f"Expected AudioBuffer, got {type(buffer)}")

        # Infer format from path if auto
        if format == "auto":
            if path.endswith(".wav"):
                format = "wav"
            elif path.endswith(".flac"):
                format = "flac"
            else:
                format = "wav"  # Default to WAV

        format = format.lower()

        # Prepare data
        # Ensure data is in the correct format (float32, clipped to [-1, 1])
        data = np.clip(buffer.data, -1.0, 1.0).astype(np.float32)

        if format == "flac":
            # FLAC requires soundfile
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError(
                    "soundfile is required for FLAC export. "
                    "Install with: pip install soundfile"
                )

            # soundfile expects (samples, channels) for stereo
            if buffer.is_stereo:
                sf.write(path, data, buffer.sample_rate, format='FLAC')
            else:
                sf.write(path, data.reshape(-1, 1), buffer.sample_rate, format='FLAC')

        elif format == "wav":
            # Try soundfile first (better quality), fall back to scipy
            try:
                import soundfile as sf
                if buffer.is_stereo:
                    sf.write(path, data, buffer.sample_rate, format='WAV')
                else:
                    sf.write(path, data.reshape(-1, 1), buffer.sample_rate, format='WAV')
            except ImportError:
                # Fall back to scipy.io.wavfile
                try:
                    from scipy.io import wavfile
                except ImportError:
                    raise ImportError(
                        "Either soundfile or scipy is required for WAV export. "
                        "Install with: pip install soundfile  OR  pip install scipy"
                    )

                # scipy.io.wavfile expects int16 format
                # Convert float32 [-1, 1] to int16 [-32768, 32767]
                data_int16 = (data * 32767).astype(np.int16)
                wavfile.write(path, buffer.sample_rate, data_int16)

        else:
            raise ValueError(
                f"Unsupported format: {format}. Supported: 'wav', 'flac'"
            )

        print(f"Saved audio to: {path}")

    @staticmethod
    def load(path: str) -> AudioBuffer:
        """Load audio buffer from file.

        Supports WAV and FLAC formats with automatic format detection.

        Args:
            path: Input file path

        Returns:
            Loaded audio buffer

        Raises:
            ImportError: If soundfile is not installed
            FileNotFoundError: If file doesn't exist

        Example:
            # Load audio file
            loaded = audio.load("input.wav")
            print(f"Loaded {loaded.duration:.2f}s of audio")
        """
        try:
            import soundfile as sf
        except ImportError:
            # Try scipy fallback for WAV
            if path.endswith('.wav'):
                try:
                    from scipy.io import wavfile
                    sample_rate, data = wavfile.read(path)

                    # Convert to float32 [-1, 1]
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.int32:
                        data = data.astype(np.float32) / 2147483648.0
                    elif data.dtype == np.uint8:
                        data = (data.astype(np.float32) - 128.0) / 128.0

                    # Handle stereo: scipy returns (samples, channels)
                    # We need to keep it that way
                    return AudioBuffer(data=data, sample_rate=sample_rate)
                except ImportError:
                    raise ImportError(
                        "Either soundfile or scipy is required for audio loading. "
                        "Install with: pip install soundfile  OR  pip install scipy"
                    )
            else:
                raise ImportError(
                    "soundfile is required for loading non-WAV audio. "
                    "Install with: pip install soundfile"
                )

        # Load with soundfile
        data, sample_rate = sf.read(path, dtype='float32')

        # soundfile returns (samples,) for mono, (samples, channels) for stereo
        # This matches our AudioBuffer expectations
        return AudioBuffer(data=data, sample_rate=sample_rate)

    @staticmethod
    def record(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
               channels: int = 1) -> AudioBuffer:
        """Record audio from microphone.

        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate in Hz
            channels: Number of channels (1=mono, 2=stereo)

        Returns:
            Recorded audio buffer

        Raises:
            ImportError: If sounddevice is not installed

        Example:
            # Record 3 seconds from microphone
            recording = audio.record(duration=3.0)
            audio.save(recording, "recording.wav")
        """
        if channels not in (1, 2):
            raise ValueError(f"channels must be 1 (mono) or 2 (stereo), got {channels}")

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for audio recording. "
                "Install with: pip install sounddevice"
            )

        print(f"Recording {duration}s of audio...")

        # Record audio
        data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete

        print("Recording complete!")

        # sounddevice returns (samples, channels) even for mono
        # For mono, we want (samples,) to match our convention
        if channels == 1:
            data = data.reshape(-1)

        return AudioBuffer(data=data, sample_rate=sample_rate)


# Create singleton instance for use as 'audio' namespace
audio = AudioOperations()
