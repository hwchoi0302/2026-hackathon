import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple

def estimate_spectrum(
    time_points: np.ndarray, 
    autocorrelation: np.ndarray, 
    window: str = 'hann', 
    zero_padding_factor: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts spectral information from time-domain autocorrelation signals using FFT.
    
    Args:
        time_points: 1D array of time points.
        autocorrelation: 1D array of autocorrelation values <psi(0)|psi(t)>.
        window: Type of windowing function ('hann', 'hamming', 'blackman', or None).
        zero_padding_factor: Factor by which to pad the signal to increase frequency resolution.
        
    Returns:
        frequencies: Array of frequency values.
        spectrum: Magnitude of the FFT.
    """
    N = len(autocorrelation)
    dt = time_points[1] - time_points[0] if N > 1 else 1.0
    
    # Apply windowing
    signal = np.array(autocorrelation, dtype=complex)
    if window:
        if window == 'hann':
            signal *= np.hanning(N)
        elif window == 'hamming':
            signal *= np.hamming(N)
        elif window == 'blackman':
            signal *= np.blackman(N)
            
    # Apply zero padding
    N_padded = N * zero_padding_factor
    padded_signal = np.pad(signal, (0, N_padded - N), 'constant')
    
    # Compute FFT
    spectrum = fft(padded_signal)
    frequencies = fftfreq(N_padded, d=dt)
    
    # Return frequencies and magnitude spectrum
    return frequencies, np.abs(spectrum)
