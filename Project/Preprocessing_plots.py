import wfdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pywt
from scipy.signal import butter, filtfilt, find_peaks

# ─── Colors ───────────────────────────────────────────────────────
BLUE   = "#1A73E8"
GREEN  = "#34A853"
RED    = "#EA4335"
ORANGE = "#FBBC05"
PURPLE = "#7B2D8B"
GRAY   = "#888888"
LIGHT  = "#F5F5F5"
DARK   = "#1C1C2E"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   LIGHT,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ─── Signal Processing Functions ─────────────────────────────────
def apply_bandpass_filter(signal, fs=500):
    nyq = 0.5 * fs
    b, a = butter(4, [1.0/nyq, 40.0/nyq], btype='band')
    return filtfilt(b, a, signal)


def pan_tompkins_r_peaks(signal, fs=500):
    diff_sig = np.zeros_like(signal)
    diff_sig[1:] = np.diff(signal)
    sq_sig = diff_sig ** 2
    window_len = int(0.15 * fs)
    integrated_sig = np.convolve(sq_sig, np.ones(window_len)/window_len, mode='same')
    threshold = np.mean(integrated_sig) + 1.5 * np.std(integrated_sig)
    distance  = int(0.3 * fs)
    peaks, _  = find_peaks(integrated_sig, height=threshold, distance=distance)
    r_peaks = []
    for p in peaks:
        start = max(0, p - int(0.05*fs))
        end   = min(len(signal), p + int(0.05*fs))
        r_peaks.append(start + np.argmax(signal[start:end]))
    return np.array(r_peaks), integrated_sig, threshold


def extract_fiducial_points(signal, r_peak, fs=500):
    q_s  = max(0, r_peak - int(0.1*fs))
    q    = q_s + np.argmin(signal[q_s:r_peak])
    s_e  = min(len(signal), r_peak + int(0.1*fs))
    s    = r_peak + np.argmin(signal[r_peak:s_e])
    pts  = {'R': r_peak, 'Q': q, 'S': s, 'P': None, 'T': None}

    qrs_onset = q - int(0.02*fs)
    p_start   = max(0, qrs_onset - int(0.200*fs))
    if p_start < qrs_onset:
        pts['P'] = p_start + np.argmax(signal[p_start:qrs_onset])

    qrs_off = s + int(0.02*fs)
    t_end   = min(len(signal), qrs_off + int(0.400*fs))
    if qrs_off < t_end:
        seg = signal[qrs_off:t_end]
        W   = np.zeros(len(seg))
        k   = 16
        for i in range(k, len(seg)-k):
            W[i] = (seg[i-k]-seg[i]) * (seg[i]-seg[i+k])
        if len(W) > 2*k:
            pts['T'] = qrs_off + k + np.argmax(W[k:-k])
    return pts


# ─── Load Data ───────────────────────────────────────────────────
try:
    RECORD_PATH  = 'data/Person_1/rec_1'
    record       = wfdb.rdrecord(RECORD_PATH)
    fs           = record.fs
    raw_signal   = record.p_signal[:fs*5, 0]   # 5 seconds
    filt_signal  = apply_bandpass_filter(raw_signal, fs)
    r_peaks, integrated_sig, threshold_val = pan_tompkins_r_peaks(filt_signal, fs)
    t            = np.arange(len(raw_signal)) / fs

    # Pick a clean single beat
    before, after = int(0.2*fs), int(0.4*fs)
    beat = None
    r_in_beat = before
    for r in r_peaks:
        if r - before >= 0 and r + after < len(filt_signal):
            beat = filt_signal[r-before: r+after]
            break

    # ══════════════════════════════════════════════════════════════
    # WINDOW 1 – Raw & Filtered Signal
    # ══════════════════════════════════════════════════════════════
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig1.suptitle("Preprocessing – Step 1 & 2", fontsize=14, fontweight='bold')

    ax1.plot(t, raw_signal, color=GRAY, lw=0.9)
    ax1.set_title("Step 1: Raw ECG Signal")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude (mV)")

    ax2.plot(t, filt_signal, color=BLUE, lw=0.9)
    ax2.set_title("Step 2: After Bandpass Filter  (1–40 Hz, 4th-order Butterworth)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Amplitude (mV)")

    fig1.canvas.manager.set_window_title('Step 1 & 2 – Raw & Filtered')
    plt.tight_layout()

    # ══════════════════════════════════════════════════════════════
    # WINDOW 2 – Differentiation & Squaring / Integration
    # ══════════════════════════════════════════════════════════════
    diff_sig = np.zeros_like(filt_signal)
    diff_sig[1:] = np.diff(filt_signal)

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 6))
    fig2.suptitle("Preprocessing – Step 3 & 4", fontsize=14, fontweight='bold')

    ax3.plot(t, diff_sig, color=ORANGE, lw=0.8)
    ax3.set_title("Step 3: Differentiation  (highlights rapid slope changes / QRS)")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("dV/dt")

    ax4.plot(t, integrated_sig, color=ORANGE, lw=1.0, label='Energy Envelope')
    ax4.axhline(threshold_val, color=RED, lw=1.4, ls='--',
                label=f'Adaptive Threshold = mean + 1.5×std')
    all_p, _ = find_peaks(integrated_sig, distance=int(0.3*fs))
    rej_p    = [p for p in all_p if integrated_sig[p] < threshold_val]
    ax4.scatter(r_peaks/fs,    integrated_sig[r_peaks], color=GREEN,
                zorder=5, s=60, label='Accepted Peaks')
    ax4.scatter([p/fs for p in rej_p], integrated_sig[rej_p], color='black',
                marker='x', s=60, label='Rejected Peaks')
    ax4.set_title("Step 4: Squaring + Moving-Average Integration & Adaptive Thresholding")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Energy")
    ax4.legend(fontsize=9)

    fig2.canvas.manager.set_window_title('Step 3 & 4 – Differentiation & Thresholding')
    plt.tight_layout()

    # ══════════════════════════════════════════════════════════════
    # WINDOW 3 – R-Peak Detection & Beat Segmentation
    # ══════════════════════════════════════════════════════════════
    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 6))
    fig3.suptitle("Preprocessing – Step 5 & 6", fontsize=14, fontweight='bold')

    ax5.plot(t, filt_signal, color=BLUE, lw=0.8, alpha=0.8, label='Filtered ECG')
    ax5.scatter(r_peaks/fs, filt_signal[r_peaks], color=RED,
                zorder=5, s=70, label='Detected R-peaks')
    ax5.set_title("Step 5: R-Peak Detection (Pan-Tompkins Algorithm)")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Amplitude (mV)")
    ax5.legend(fontsize=9)

    if beat is not None:
        beat_t = (np.arange(len(beat)) - r_in_beat) / fs * 1000
        ax6.plot(beat_t, beat, color=GREEN, lw=1.8)
        ax6.axvline(0, color=RED, lw=1.2, ls='--', label='R-peak  (t = 0 ms)')
        ax6.axvspan(-200, 0,   alpha=0.08, color=BLUE,   label='Pre-R  (−200 ms)')
        ax6.axvspan(0,    400, alpha=0.08, color=ORANGE, label='Post-R (+400 ms)')
        ax6.set_title("Step 6: Single Heartbeat Segmentation  (−200 ms to +400 ms around R)")
        ax6.set_xlabel("Time relative to R-peak (ms)"); ax6.set_ylabel("Amplitude (mV)")
        ax6.legend(fontsize=9)

    fig3.canvas.manager.set_window_title('Step 5 & 6 – R-peaks & Segmentation')
    plt.tight_layout()

    # ══════════════════════════════════════════════════════════════
    # WINDOW 4 – Fiducial Points on Single Beat
    # ══════════════════════════════════════════════════════════════
    if beat is not None:
        pts = None
        for r in r_peaks:
            if r - before >= 0 and r + after < len(filt_signal):
                b   = filt_signal[r-before: r+after]
                p   = extract_fiducial_points(b, before, fs)
                if all(v is not None for v in p.values()):
                    beat_fid, pts = b, p
                    break

        if pts:
            beat_t = (np.arange(len(beat_fid)) - before) / fs * 1000
            fig4, ax7 = plt.subplots(figsize=(11, 5))

            ax7.plot(beat_t, beat_fid, color=BLUE, lw=2, label='ECG Beat')

            pt_colors = {'P': ORANGE, 'Q': GREEN, 'R': RED, 'S': PURPLE, 'T': BLUE}
            pt_labels = {
                'P': 'P-wave peak',
                'Q': 'Q-point  (QRS onset)',
                'R': 'R-peak',
                'S': 'S-point  (QRS offset)',
                'T': 'T-wave peak'
            }
            for name, idx in pts.items():
                if idx is not None and 0 <= idx < len(beat_fid):
                    x = beat_t[idx]; y = beat_fid[idx]
                    ax7.scatter(x, y, color=pt_colors[name], zorder=6, s=110)
                    ax7.annotate(
                        pt_labels[name], xy=(x, y),
                        xytext=(x+18, y + 0.045), fontsize=9,
                        color=pt_colors[name],
                        arrowprops=dict(arrowstyle='->', color=pt_colors[name], lw=1.2)
                    )

            # Interval arrows
            if pts['P'] and pts['R']:
                px = beat_t[pts['P']]; rx = beat_t[pts['R']]
                ax7.annotate('', xy=(rx, -0.13), xytext=(px, -0.13),
                             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                ax7.text((px+rx)/2, -0.155, 'PR Interval',
                         ha='center', fontsize=8.5, color='black')

            if pts['Q'] and pts['S']:
                qx = beat_t[pts['Q']]; sx = beat_t[pts['S']]
                ax7.annotate('', xy=(sx, -0.19), xytext=(qx, -0.19),
                             arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5))
                ax7.text((qx+sx)/2, -0.215, 'QRS Duration',
                         ha='center', fontsize=8.5, color=GREEN)

            ax7.set_title("Feature Extraction – Fiducial Points (P, Q, R, S, T) on Single ECG Beat",
                          fontsize=13, fontweight='bold')
            ax7.set_xlabel("Time relative to R-peak (ms)")
            ax7.set_ylabel("Amplitude (mV)")
            ax7.legend(fontsize=9, loc='upper right')
            fig4.canvas.manager.set_window_title('Fiducial Points – P Q R S T')
            plt.tight_layout()

    # ══════════════════════════════════════════════════════════════
    # WINDOW 5 – Wavelet Decomposition (db4, level 4)
    # ══════════════════════════════════════════════════════════════
    if beat is not None:
        coeffs = pywt.wavedec(beat, 'db4', level=4)
        titles = [
            'cA4 – Approximation  (Low Freq, 0–15 Hz)   ✓ USED',
            'cD4 – Detail Level 4  (15–30 Hz)            ✓ USED',
            'cD3 – Detail Level 3  (30–60 Hz)            ✓ USED',
            'cD2 – Detail Level 2  (60–125 Hz)',
            'cD1 – Detail Level 1  (125–250 Hz)',
        ]
        clrs  = [BLUE, GREEN, ORANGE, GRAY, GRAY]
        used  = [True, True, True, False, False]

        fig5, axes = plt.subplots(len(coeffs), 1, figsize=(12, 12))
        fig5.suptitle("Feature Extraction – Wavelet Decomposition  (db4, Level 4)\n"
                      "Features used: Mean, Std Dev, Energy  of  cA4, cD4, cD3  →  9 values",
                      fontsize=13, fontweight='bold')

        for i, (c, ax) in enumerate(zip(coeffs, axes)):
            ax.plot(c, color=clrs[i], lw=1.3)
            ax.set_title(titles[i], fontsize=10)
            ax.set_ylabel("Amplitude")
            stats = (f"Mean = {np.mean(c):.4f}    "
                     f"Std = {np.std(c):.4f}    "
                     f"Energy = {np.sum(c**2):.4f}")
            fc = '#F0FFF4' if used[i] else LIGHT
            ax.set_facecolor(fc)
            ax.text(0.02, 0.82, stats, transform=ax.transAxes, fontsize=8.5,
                    color='#333333',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='#AAAAAA',
                              boxstyle='round,pad=0.3'))

        fig5.canvas.manager.set_window_title('Wavelet Decomposition – db4 Level 4')
        plt.tight_layout()

    # Show all windows
    plt.show()

except Exception as e:
    print(f"Error: {e}")
    raise