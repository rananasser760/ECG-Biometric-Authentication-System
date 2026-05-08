import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, find_peaks


# --- 1. Signal Processing Functions ---

def apply_bandpass_filter(signal, fs=500):
    nyq = 0.5 * fs
    b, a = butter(4, [1.0 / nyq, 40.0 / nyq], btype='band')
    return filtfilt(b, a, signal)


def pan_tompkins_r_peaks(signal, fs=500):
    diff_sig = np.zeros_like(signal)
    diff_sig[1:] = np.diff(signal)
    sq_sig = diff_sig ** 2
    window_len = int(0.15 * fs)
    integrated_sig = np.convolve(sq_sig, np.ones(window_len) / window_len, mode='same')

    threshold = np.mean(integrated_sig) + 1.5 * np.std(integrated_sig)
    distance = int(0.3 * fs)
    peaks, _ = find_peaks(integrated_sig, height=threshold, distance=distance)

    r_peaks = []
    for p in peaks:
        start = max(0, p - int(0.05 * fs))
        end = min(len(signal), p + int(0.05 * fs))
        r_peak = start + np.argmax(signal[start:end])
        r_peaks.append(r_peak)

    return np.array(r_peaks), integrated_sig, threshold


# --- 2. Processing Data ---

try:
    RECORD_PATH = 'data/Person_1/rec_1'
    record = wfdb.rdrecord(RECORD_PATH)
    fs = record.fs
    raw_signal = record.p_signal[:fs * 5, 0]  # Use 5 seconds

    filtered_signal = apply_bandpass_filter(raw_signal, fs)
    r_peaks, integrated_sig, threshold_val = pan_tompkins_r_peaks(filtered_signal, fs)

    beat = None
    if len(r_peaks) > 0:
        r = r_peaks[0]
        beat = filtered_signal[r - int(0.2 * fs): r + int(0.4 * fs)]

    # --- 3. Optimized Plotting (Grouped as requested) ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # WINDOW 1: Raw and Filtered (Together)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(raw_signal, color='gray')
    ax1.set_title("Step 1: Raw ECG Signal")
    ax2.plot(filtered_signal, color='blue')
    ax2.set_title("Step 2: Signal After Bandpass Filter (1-40 Hz)")
    fig1.canvas.manager.set_window_title('Preprocessing: Raw & Filtered')
    plt.tight_layout()

    # WINDOW 2: Threshold Logic and Final R-Peaks (Together)
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    # Threshold Logic
    ax3.plot(integrated_sig, color='orange', label='Energy Envelope')
    ax3.axhline(y=threshold_val, color='red', linestyle='--', label='Threshold')
    all_p, _ = find_peaks(integrated_sig, distance=int(0.3 * fs))
    rej_p = [p for p in all_p if integrated_sig[p] < threshold_val]
    ax3.scatter(r_peaks, integrated_sig[r_peaks], color='green', label='Accepted')
    ax3.scatter(rej_p, integrated_sig[rej_p], color='black', marker='x', label='Rejected')
    ax3.set_title("Step 3 & 4: Adaptive Thresholding Logic")
    ax3.legend()
    # Final R-Peaks
    ax4.plot(filtered_signal, color='blue', alpha=0.7)
    ax4.scatter(r_peaks, filtered_signal[r_peaks], color='red', label='Detected R-peaks')
    ax4.set_title("Step 5: Final R-peaks Detection")
    ax4.legend()
    fig2.canvas.manager.set_window_title('Detection Logic: Threshold & R-peaks')
    plt.tight_layout()

    # WINDOW 3: Single Segmented Beat (Alone)
    if beat is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(beat, color='green')
        plt.title("Step 6: Single Segmented Heartbeat")
        plt.gcf().canvas.manager.set_window_title('Segmentation: Single Beat')
        plt.tight_layout()

    # WINDOW 4: Wavelet Coefficients (Together)
    if beat is not None:
        coeffs = pywt.wavedec(beat, 'db4', level=4)
        fig4, axes = plt.subplots(len(coeffs), 1, figsize=(8, 10))
        titles = ['cA4 (Approx)', 'cD4', 'cD3', 'cD2', 'cD1']
        for i, (c, t) in enumerate(zip(coeffs, titles)):
            axes[i].plot(c, color='purple')
            axes[i].set_title(f"Step 7: Wavelet Coefficient - {t}")
        fig4.canvas.manager.set_window_title('Features: Wavelet Coefficients')
        plt.tight_layout()

    # عرض جميع النوافذ في وقت واحد
    plt.show()

except Exception as e:
    print(f"Error: {e}")