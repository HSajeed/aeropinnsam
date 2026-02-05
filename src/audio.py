"""
Auralization Module for Airfoil Noise

This module converts Sound Pressure Level (SPL) predictions into
audible sound, enabling intuitive interpretation of model outputs.

The auralization process:
1. Convert SPL (dB) to pressure (Pa) using reference pressure
2. Create frequency spectrum based on predicted values
3. Synthesize audio signal using spectral synthesis
"""

import numpy as np
import soundfile as sf
from typing import Optional


# Reference pressure for SPL calculations (threshold of hearing)
P_REF = 20e-6  # 20 micropascals


def spl_to_pressure(spl: np.ndarray) -> np.ndarray:
    """
    Convert Sound Pressure Level (dB) to pressure (Pa).
    
    SPL = 20 * log10(p / p_ref)
    Therefore: p = p_ref * 10^(SPL/20)
    
    Args:
        spl: Sound Pressure Level in dB
        
    Returns:
        Pressure in Pascals
    """
    return P_REF * np.power(10, spl / 20)


def generate_airfoil_sound(
    predicted_spl_spectrum: np.ndarray,
    frequencies: np.ndarray,
    duration: float = 2.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Synthesize airfoil noise from predicted SPL spectrum.
    
    Uses spectral synthesis to create an audio signal that
    matches the predicted frequency-SPL characteristics.
    
    Args:
        predicted_spl_spectrum: SPL values at each frequency (dB)
        frequencies: Corresponding frequency values (Hz)
        duration: Audio duration in seconds
        sample_rate: Sampling rate in Hz
        
    Returns:
        Synthesized audio signal (normalized)
    """
    # Create time vector
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Convert SPL to pressure
    pressure_spectrum = spl_to_pressure(predicted_spl_spectrum)
    
    # Initialize audio signal
    audio_signal = np.zeros_like(t)
    
    # Synthesize using additive synthesis with random phases
    for freq, pressure in zip(frequencies, pressure_spectrum):
        # Random phase for natural sound
        phase = np.random.uniform(0, 2 * np.pi)
        audio_signal += pressure * np.sin(2 * np.pi * freq * t + phase)
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(audio_signal))
    if max_amplitude > 0:
        audio_signal = audio_signal / max_amplitude
    
    return audio_signal


def create_spectrum_from_spl(
    predicted_spl: float,
    freq_min: float = 200,
    freq_max: float = 5000,
    num_freqs: int = 50
) -> tuple:
    """
    Create a spectrum shape from a single SPL prediction.
    
    In a full implementation, the model would predict the entire
    spectrum. Here we approximate by assuming the spectrum falls
    off with frequency following typical broadband noise characteristics.
    
    Args:
        predicted_spl: Single SPL prediction (dB)
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
        num_freqs: Number of frequency bins
        
    Returns:
        Tuple of (frequencies, spl_spectrum)
    """
    frequencies = np.linspace(freq_min, freq_max, num_freqs)
    
    # Approximate broadband spectrum shape
    # SPL typically drops with log(frequency)
    spectrum_shape = predicted_spl - 10 * np.log10(frequencies / 1000)
    
    return frequencies, spectrum_shape


def save_audio(
    audio_signal: np.ndarray,
    filepath: str,
    sample_rate: int = 44100
) -> None:
    """
    Save audio signal to file.
    
    Args:
        audio_signal: Audio signal to save
        filepath: Output file path (e.g., 'output.wav')
        sample_rate: Sampling rate in Hz
    """
    sf.write(filepath, audio_signal, sample_rate)


def auralize_prediction(
    predicted_spl: float,
    output_path: str = 'airfoil_noise.wav',
    duration: float = 2.0,
    sample_rate: int = 44100
) -> str:
    """
    Complete auralization pipeline: SPL prediction to audio file.
    
    Args:
        predicted_spl: Predicted SPL value (dB)
        output_path: Path for output audio file
        duration: Audio duration in seconds
        sample_rate: Sampling rate in Hz
        
    Returns:
        Path to saved audio file
    """
    # Create spectrum
    frequencies, spectrum = create_spectrum_from_spl(predicted_spl)
    
    # Generate audio
    audio = generate_airfoil_sound(
        spectrum, 
        frequencies, 
        duration=duration, 
        sample_rate=sample_rate
    )
    
    # Save to file
    save_audio(audio, output_path, sample_rate)
    
    return output_path
