import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def perform_fft_analysis(
    times: np.ndarray,
    expectation_values: np.ndarray,
    window_type: str = "hann",
    zero_padding_factor: int = 4
) -> dict:
    """
    Performs FFT spectral analysis on time-dependent expectation values.
    
    Args:
        times: 1D array of time points.
        expectation_values: 1D array of expectation values over time.
        window_type: 'hann', 'hamming', 'blackman', or None.
        zero_padding_factor: Padding multiplier to increase frequency resolution.
        
    Returns:
        dict containing:
            - frequencies: Frequency bins.
            - spectrum: Absolute magnitude of the FFT spectrum.
            - peak_frequencies: Frequencies where local maxima occur.
            - peak_values: Magnitudes at the local maxima.
    """
    N = len(expectation_values)
    if N <= 1:
        raise ValueError("Signal must have at least 2 points to perform FFT.")
        
    dt = times[1] - times[0]
    
    # 1. Detrend / subtract mean to eliminate large DC component at f=0
    signal = np.array(expectation_values) - np.mean(expectation_values)
    
    # 2. Apply windowing
    if window_type == "hann":
        signal *= np.hanning(N)
    elif window_type == "hamming":
        signal *= np.hamming(N)
    elif window_type == "blackman":
        signal *= np.blackman(N)
    elif window_type is not None:
        raise ValueError(f"Unknown window type: {window_type}")
        
    # 3. Zero padding
    N_padded = N * zero_padding_factor
    padded_signal = np.pad(signal, (0, N_padded - N), 'constant')
    
    # 4. Compute FFT
    fft_vals = fft(padded_signal)
    freqs = fftfreq(N_padded, d=dt)
    
    # Only keep positive frequencies
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    spectrum = np.abs(fft_vals)[positive_mask]
    
    # 5. Find peaks (local maxima)
    max_val = np.max(spectrum)
    threshold = 0.05 * max_val if max_val > 0 else 0.0
    peaks, _ = find_peaks(spectrum, height=threshold)
    
    # Sort peaks by height descending
    peak_heights = spectrum[peaks]
    sorted_idx = np.argsort(peak_heights)[::-1]
    sorted_peaks = peaks[sorted_idx]
    
    return {
        "frequencies": freqs,
        "spectrum": spectrum,
        "peak_frequencies": freqs[sorted_peaks],
        "peak_values": spectrum[sorted_peaks]
    }
