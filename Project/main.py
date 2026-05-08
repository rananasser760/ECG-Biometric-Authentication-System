import os
import wfdb
import numpy as np
import pywt
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import threading
import math

# ─────────────────────────────────────────────
#  Signal Processing & Fiducial Points Extraction
# ─────────────────────────────────────────────

def apply_bandpass_filter(signal, fs=500):
    nyq = 0.5 * fs
    low = 1.0 / nyq
    high = 40.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)


def pan_tompkins_r_peaks(signal, fs=500):
    # 1. Differentiation: Highlight rapid changes in the ECG signal
    # Initialize with zeros to maintain original signal length and avoid indexing issues
    diff_sig = np.zeros_like(signal)
    diff_sig[1:] = np.diff(signal)

    # 2. Squaring: Amplify high-frequency components (QRS complex) and suppress noise
    sq_sig = diff_sig ** 2

    # 3. Moving Average Integration: Smooth the signal to obtain the energy envelope
    window_len = int(0.15 * fs)
    integrated_sig = np.convolve(sq_sig, np.ones(window_len) / window_len, mode='same')

    # 4. Adaptive Thresholding: Calculate a dynamic threshold based on signal energy
    # Using (Mean + 1.5 * STD) to distinguish R-peaks from T-waves and noise
    threshold = np.mean(integrated_sig) + 1.5 * np.std(integrated_sig)

    # 5. Peak Detection: Find candidate peaks in the integrated signal
    # 'distance' ensures at least 0.3s between beats, supporting heart rates up to 200 BPM
    distance = int(0.3 * fs)
    peaks, _ = find_peaks(integrated_sig, height=threshold, distance=distance)

    # 6. R-peak Refinement: Locate the exact R-peak in the original filtered signal
    # Search within a small window around the energy peak for maximum amplitude
    r_peaks = []
    for p in peaks:
        start = max(0, p - int(0.05 * fs))
        end = min(len(signal), p + int(0.05 * fs))
        r_peak = start + np.argmax(signal[start:end])
        r_peaks.append(r_peak)

    return np.array(r_peaks)
def extract_heartbeats(signal, fs=500):
    r_peaks = pan_tompkins_r_peaks(signal, fs)
    heartbeats = []
    r_peaks_in_beats = []
    before = int(0.2 * fs)
    after  = int(0.4 * fs)
    for r_peak in r_peaks:
        if r_peak - before >= 0 and r_peak + after < len(signal):
            heartbeats.append(signal[r_peak - before: r_peak + after])
            r_peaks_in_beats.append(before)
    return heartbeats, r_peaks_in_beats

def get_qrs_onset_offset(signal, r_peak, fs=500):
    search_q_start = max(0, r_peak - int(0.1 * fs))
    q_peak = search_q_start + np.argmin(signal[search_q_start:r_peak]) if search_q_start < r_peak else r_peak
    search_s_end = min(len(signal), r_peak + int(0.1 * fs))
    s_peak = r_peak + np.argmin(signal[r_peak:search_s_end]) if r_peak < search_s_end else r_peak
    return q_peak, s_peak

def extract_fiducial_points(signal, r_peak, fs=500):
    q_peak, s_peak = get_qrs_onset_offset(signal, r_peak, fs)
    points = {'R': r_peak, 'Q': q_peak, 'S': s_peak, 'P': None, 'T': None}

    p_window_width = int(0.200 * fs)
    qrs_onset = q_peak - int(0.02 * fs)
    search_p_start = max(0, qrs_onset - p_window_width)
    if search_p_start < qrs_onset:
        points['P'] = search_p_start + np.argmax(signal[search_p_start:qrs_onset])

    t_window_width = int(0.400 * fs)
    qrs_offset = s_peak + int(0.02 * fs)
    search_t_end = min(len(signal), qrs_offset + t_window_width)
    if qrs_offset < search_t_end:
        window_signal = signal[qrs_offset:search_t_end]
        W = np.zeros_like(window_signal)
        k = 16
        for i in range(k, len(window_signal) - k):
            W1 = window_signal[i - k] - window_signal[i]
            W2 = window_signal[i] - window_signal[i + k]
            W[i] = W1 * W2
        if len(W) > 2 * k:
            t_rel_idx = k + np.argmax(W[k:-k])
            points['T'] = qrs_offset + t_rel_idx

    return points

def get_combined_features(heartbeats, r_peaks_in_beats, wavelet_name, fs=500):
    features = []
    for beat, r_peak in zip(heartbeats, r_peaks_in_beats):
        coeffs = pywt.wavedec(beat, wavelet_name, level=4 ) #A4: (0:15) ,, D4 : (15:30) ,, D3 : (30:60)
        selected_coeffs = coeffs[:3]
        beat_features = []
        for c in selected_coeffs:
            beat_features.extend([np.mean(c), np.std(c), np.sum(np.square(c))])

        points = extract_fiducial_points(beat, r_peak, fs)
        if points and points['R'] is not None and points['Q'] is not None and points['S'] is not None:
            pr_interval  = (points['R'] - points['P']) / fs if points['P'] is not None else 0
            qt_interval  = (points['T'] - points['Q']) / fs if points['T'] is not None else 0
            qrs_duration = (points['S'] - points['Q']) / fs
            beat_features.extend([pr_interval, qt_interval, qrs_duration])
        else:
            beat_features.extend([0, 0, 0])

        features.append(beat_features)
    return np.array(features)

def load_and_prepare_data(base_path):
    X_train_raw, y_train = [], []
    X_test_raw,  y_test  = [], []
    R_train, R_test      = [], []
    for person_id in range(1, 6):
        person_folder = os.path.join(base_path, f'Person_{person_id}')
        if not os.path.exists(person_folder):
            continue
        files = [f.split('.')[0] for f in os.listdir(person_folder) if f.endswith('.dat')]
        files.sort()
        train_files = files[:5]
        test_files  = files[-2:] if len(files) >= 7 else files[5:]
        for file in train_files:
            record_path = os.path.join(person_folder, file)
            record = wfdb.rdrecord(record_path)
            signal = apply_bandpass_filter(record.p_signal[:, 0])
            beats, r_locs = extract_heartbeats(signal)
            X_train_raw.extend(beats)
            R_train.extend(r_locs)
            y_train.extend([person_id] * len(beats))
        for file in test_files:
            record_path = os.path.join(person_folder, file)
            record = wfdb.rdrecord(record_path)
            signal = apply_bandpass_filter(record.p_signal[:, 0])
            beats, r_locs = extract_heartbeats(signal)
            X_test_raw.extend(beats)
            R_test.extend(r_locs)
            y_test.extend([person_id] * len(beats))
    return X_train_raw, R_train, np.array(y_train), X_test_raw, R_test, np.array(y_test)

# ─────────────────────────────────────────────
#  Model Training
# ─────────────────────────────────────────────
base_data_path = 'data'
X_train_raw, R_train, y_train, X_test_raw, R_test, y_test = load_and_prepare_data(base_data_path)

wavelets = ['db1', 'db2', 'db4']

# 3 parameter sets per classifier
svm_params = [
    {'kernel': 'rbf',    'C': 1,   'gamma': 'scale'},
    {'kernel': 'rbf',    'C': 10,  'gamma': 'scale'},
    {'kernel': 'linear', 'C': 1},
]
rf_params = [
    {'n_estimators': 50,  'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': None},
]
knn_params = [
    {'n_neighbors': 3, 'weights': 'uniform'},
    {'n_neighbors': 5, 'weights': 'uniform'},
    {'n_neighbors': 7, 'weights': 'uniform'},
]

best_overall_acc = 0.0
best_model       = None
best_scaler      = None
best_wavelet     = 'db4'

wavelet_results  = {}

print("=" * 75)
print("  Parameter Tuning per Classifier & Wavelet")
print("=" * 75)

for wv in wavelets:
    X_train = get_combined_features(X_train_raw, R_train, wv)
    X_test  = get_combined_features(X_test_raw,  R_test,  wv)
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\n  Wavelet: {wv}")
    print("  " + "─" * 70)
    print(f"  {'Classifier':<16} {'Parameters':<35} {'Accuracy':>8}")
    print("  " + "─" * 70)

    wv_best = {}

    # ── SVM ──────────────────────────────────────────
    best_svm_acc = 0
    best_svm     = None
    for p in svm_params:
        clf = SVC(**p, probability=True)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
        tag = '  ◄ best' if acc > best_svm_acc else ''
        param_str = ', '.join(f"{k}={v}" for k, v in p.items())
        print(f"  {'SVM':<16} {param_str:<35} {acc:>7.2f}%{tag}")
        if acc > best_svm_acc:
            best_svm_acc = acc
            best_svm     = clf
    wv_best['SVM'] = (best_svm, best_svm_acc)

    # ── Random Forest ────────────────────────────────
    best_rf_acc = 0
    best_rf     = None
    for p in rf_params:
        clf = RandomForestClassifier(**p, random_state=42)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
        tag = '  ◄ best' if acc > best_rf_acc else ''
        param_str = ', '.join(f"{k}={v}" for k, v in p.items())
        print(f"  {'Random Forest':<16} {param_str:<35} {acc:>7.2f}%{tag}")
        if acc > best_rf_acc:
            best_rf_acc = acc
            best_rf     = clf
    wv_best['Random Forest'] = (best_rf, best_rf_acc)

    # ── KNN ──────────────────────────────────────────
    best_knn_acc = 0
    best_knn     = None
    for p in knn_params:
        clf = KNeighborsClassifier(**p)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
        tag = '  ◄ best' if acc > best_knn_acc else ''
        param_str = ', '.join(f"{k}={v}" for k, v in p.items())
        print(f"  {'KNN':<16} {param_str:<35} {acc:>7.2f}%{tag}")
        if acc > best_knn_acc:
            best_knn_acc = acc
            best_knn     = clf
    wv_best['KNN'] = (best_knn, best_knn_acc)

    wavelet_results[wv] = (wv_best, scaler)

    # update overall best model
    for clf_name, (clf, acc) in wv_best.items():
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_model       = clf
            best_scaler      = scaler
            best_wavelet     = wv

# ─────────────────────────────────────────────
#  Compare Best Result per Classifier
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  Best Result per Classifier (across all wavelets)")
print("=" * 75)
print(f"  {'Classifier':<16} {'Best Wavelet':<14} {'Accuracy':>8}")
print("  " + "─" * 40)

for clf_name in ['SVM', 'Random Forest', 'KNN']:
    top_acc = 0
    top_wv  = ''
    for wv, (wv_best, _) in wavelet_results.items():
        _, acc = wv_best[clf_name]
        if acc > top_acc:
            top_acc = acc
            top_wv  = wv
    print(f"  {clf_name:<16} {top_wv:<14} {top_acc:>7.2f}%")

print("=" * 75)

clf_display_names = {'SVC': 'SVM', 'RandomForestClassifier': 'Random Forest', 'KNeighborsClassifier': 'KNN'}
best_clf_name = clf_display_names.get(type(best_model).__name__, type(best_model).__name__)
print(f"\n  ★ Overall Best: {best_clf_name:<16} | Wavelet: {best_wavelet} | Accuracy: {best_overall_acc:.2f}%\n")

# ─────────────────────────────────────────────
#  UI Design & Styling
# ─────────────────────────────────────────────

COLORS = {
    'bg_dark':    '#0A0E1A',
    'bg_panel':   '#0F1628',
    'bg_card':    '#141D35',
    'accent':     '#00D4FF',
    'accent2':    '#0099CC',
    'success':    '#00FF9C',
    'warning':    '#FFB800',
    'danger':     '#FF4D6D',
    'text_main':  '#E8F0FF',
    'text_dim':   '#6B7A99',
    'border':     '#1E2D50',
    'highlight':  '#1A2845',
}

FONTS = {
    'title':    ('Courier New', 22, 'bold'),
    'subtitle': ('Courier New', 10),
    'label':    ('Courier New', 11, 'bold'),
    'body':     ('Courier New', 10),
    'small':    ('Courier New', 9),
    'result':   ('Courier New', 14, 'bold'),
    'big':      ('Courier New', 28, 'bold'),
}

def draw_ecg_line(canvas, width, height, color, offset=0):
    canvas.delete("ecg")
    pts = []
    x = 0
    step = 4
    while x < width:
        phase = (x + offset) % 120
        if phase < 40:
            y = height // 2
        elif phase < 45:
            y = height // 2 + 6
        elif phase < 50:
            y = height // 2 - 28
        elif phase < 55:
            y = height // 2 + 12
        elif phase < 60:
            y = height // 2 - 6
        elif phase < 80:
            y = height // 2 - int(8 * math.sin(math.pi * (phase - 60) / 20))
        else:
            y = height // 2
        pts.extend([x, y])
        x += step
    if len(pts) >= 4:
        canvas.create_line(pts, fill=color, width=2, smooth=True, tags="ecg")

class BiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG · Biometric Authentication")
        self.root.state('zoomed')
        self.root.configure(bg=COLORS['bg_dark'])
        self._ecg_offset = 0
        self._animating = False
        self._current_photo = None
        self._build_ui()
        self._start_ecg_animation()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=COLORS['bg_dark'])
        header.pack(fill='x', padx=0, pady=0)
        self.ecg_canvas = tk.Canvas(header, height=60, bg=COLORS['bg_dark'], highlightthickness=0)
        self.ecg_canvas.pack(fill='x', expand=True)
        self.ecg_canvas.bind('<Configure>', lambda e: None)
        title_bar = tk.Frame(self.root, bg=COLORS['bg_panel'], pady=0)
        title_bar.pack(fill='x')
        tk.Frame(title_bar, bg=COLORS['accent'], width=4, height=70).pack(side='left')
        title_inner = tk.Frame(title_bar, bg=COLORS['bg_panel'], padx=20)
        title_inner.pack(side='left', fill='y', pady=12)
        tk.Label(title_inner, text="E C G   B I O M E T R I C   A U T H", font=FONTS['title'], bg=COLORS['bg_panel'], fg=COLORS['accent']).pack(anchor='w')
        tk.Label(title_inner, text="WAVELET & FIDUCIAL  ·  SVM  ·  IDENTIFICATION", font=FONTS['subtitle'], bg=COLORS['bg_panel'], fg=COLORS['text_dim']).pack(anchor='w')
        status_fr = tk.Frame(title_bar, bg=COLORS['bg_panel'])
        status_fr.pack(side='right', padx=20)
        self.status_dot = tk.Canvas(status_fr, width=12, height=12, bg=COLORS['bg_panel'], highlightthickness=0)
        self.status_dot.pack(side='left')
        self.status_dot.create_oval(2, 2, 10, 10, fill=COLORS['success'], outline='')
        tk.Label(status_fr, text="SYSTEM  READY", font=FONTS['small'], bg=COLORS['bg_panel'], fg=COLORS['success']).pack(side='left', padx=6)
        tk.Frame(self.root, bg=COLORS['border'], height=1).pack(fill='x')
        body = tk.Frame(self.root, bg=COLORS['bg_dark'])
        body.pack(fill='both', expand=True, padx=28, pady=20)
        left = tk.Frame(body, bg=COLORS['bg_dark'])
        left.pack(side='left', fill='both', expand=True)
        self._card(left, "01 / SELECT  RECORD", self._build_select_section).pack(fill='x', pady=(0, 14))
        self._card(left, "02 / ANALYSIS  PARAMETERS", self._build_info_section).pack(fill='x', pady=(0, 14))
        self._card(left, "03 / IDENTIFICATION  RESULT", self._build_result_section).pack(fill='x', pady=(0, 14))
        right = tk.Frame(body, bg=COLORS['bg_dark'], width=340)
        right.pack(side='right', fill='both', padx=(20, 0))
        right.pack_propagate(False)
        self._card(right, "SUBJECT  PHOTO", self._build_photo_section, expand=True).pack(fill='both', expand=True)
        footer = tk.Frame(self.root, bg=COLORS['bg_panel'])
        footer.pack(fill='x', side='bottom')
        tk.Frame(footer, bg=COLORS['border'], height=1).pack(fill='x')
        tk.Label(footer, text="ECG-LOCK  v2.0  ·  db4 WAVELET + FIDUCIALS  ·  RBF-SVM", font=FONTS['small'], bg=COLORS['bg_panel'], fg=COLORS['text_dim']).pack(pady=8)

    def _card(self, parent, title, content_fn, expand=False):
        frame = tk.Frame(parent, bg=COLORS['bg_card'], highlightbackground=COLORS['border'], highlightthickness=1)
        hdr = tk.Frame(frame, bg=COLORS['highlight'], pady=7, padx=14)
        hdr.pack(fill='x')
        tk.Frame(hdr, bg=COLORS['accent'], width=3, height=16).pack(side='left')
        tk.Label(hdr, text=f"  {title}", font=FONTS['small'], bg=COLORS['highlight'], fg=COLORS['accent']).pack(side='left')
        body = tk.Frame(frame, bg=COLORS['bg_card'], padx=16, pady=12)
        body.pack(fill='both', expand=expand)
        content_fn(body)
        return frame

    def _build_select_section(self, parent):
        tk.Label(parent, text="Load a WFDB .dat record to authenticate.", font=FONTS['body'], bg=COLORS['bg_card'], fg=COLORS['text_dim'], wraplength=300, justify='left').pack(anchor='w', pady=(0, 10))
        btn = tk.Button(parent, text="  ▶   LOAD  ECG  RECORD", command=self._run_processing, font=FONTS['label'], bg=COLORS['accent'], fg=COLORS['bg_dark'], activebackground=COLORS['accent2'], activeforeground=COLORS['bg_dark'], relief='flat', cursor='hand2', padx=18, pady=10, bd=0)
        btn.pack(fill='x')
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(parent, variable=self.progress_var, maximum=100, mode='indeterminate', style='ECG.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=(8, 0))
        self.progress.pack_forget()
        self._style_progressbar()
        self.filepath_label = tk.Label(parent, text="No file selected.", font=FONTS['small'], bg=COLORS['bg_card'], fg=COLORS['text_dim'], anchor='w')
        self.filepath_label.pack(fill='x', pady=(6, 0))

    def _style_progressbar(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure('ECG.Horizontal.TProgressbar', troughcolor=COLORS['bg_dark'], background=COLORS['accent'], thickness=4)

    def _build_info_section(self, parent):
        rows = [
            ("Wavelet",     "Daubechies db4"),
            ("Fiducial Pts","Pan-Tompkins + Wing Func"),
            ("Classifier", f"{best_clf_name}"),
            ("Features",    "Wavelet Energy + Intervals"),
            ("Threshold", "> 80% of beats vote same subject"),
        ]
        for label, value in rows:
            row = tk.Frame(parent, bg=COLORS['bg_card'])
            row.pack(fill='x', pady=2)
            tk.Label(row, text=f"{label}:", font=FONTS['small'], bg=COLORS['bg_card'], fg=COLORS['text_dim'], width=14, anchor='w').pack(side='left')
            tk.Label(row, text=value, font=FONTS['small'], bg=COLORS['bg_card'], fg=COLORS['text_main'], anchor='w').pack(side='left')

    def _build_result_section(self, parent):
        self.result_id_label = tk.Label(parent, text="—", font=FONTS['big'], bg=COLORS['bg_card'], fg=COLORS['text_dim'])
        self.result_id_label.pack(anchor='w')
        self.result_label = tk.Label(parent, text="Awaiting ECG input...", font=FONTS['result'], bg=COLORS['bg_card'], fg=COLORS['text_dim'], wraplength=300, justify='left')
        self.result_label.pack(anchor='w', pady=(0, 8))
        conf_row = tk.Frame(parent, bg=COLORS['bg_card'])
        conf_row.pack(fill='x')
        tk.Label(conf_row, text="CONFIDENCE", font=FONTS['small'], bg=COLORS['bg_card'], fg=COLORS['text_dim']).pack(side='left')
        self.conf_label = tk.Label(conf_row, text="—", font=FONTS['small'], bg=COLORS['bg_card'], fg=COLORS['accent'])
        self.conf_label.pack(side='right')
        bar_bg = tk.Frame(parent, bg=COLORS['bg_dark'], height=6)
        bar_bg.pack(fill='x', pady=(4, 0))
        bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_bg, bg=COLORS['text_dim'], height=6, width=0)
        self.conf_bar.place(x=0, y=0, height=6)
        self._bar_total_width = 0
        def _store_width(event):
            self._bar_total_width = event.width
        bar_bg.bind('<Configure>', _store_width)

    def _build_photo_section(self, parent):
        self.photo_frame = tk.Frame(parent, bg=COLORS['bg_card'])
        self.photo_frame.pack(fill='both', expand=True)
        self.photo_canvas = tk.Canvas(self.photo_frame, width=300, height=320, bg=COLORS['bg_dark'], highlightbackground=COLORS['border'], highlightthickness=1)
        self.photo_canvas.pack(fill='both', expand=True)
        self._draw_placeholder()
        self.photo_name_label = tk.Label(self.photo_frame, text="UNKNOWN", font=FONTS['label'], bg=COLORS['bg_card'], fg=COLORS['text_dim'])
        self.photo_name_label.pack(pady=(10, 0))

    def _draw_placeholder(self):
        self.photo_canvas.delete("all")
        c = self.photo_canvas
        w, h = 300, 320
        for i in range(0, w, 20):
            c.create_line(i, 0, i, h, fill='#1A2235', width=1)
        for i in range(0, h, 20):
            c.create_line(0, i, w, i, fill='#1A2235', width=1)
        c.create_oval(110, 60, 190, 140, outline=COLORS['border'], width=2)
        c.create_arc(60, 130, 240, 260, start=0, extent=180, outline=COLORS['border'], width=2, style='arc')
        c.create_text(150, 280, text="NO  SUBJECT", fill=COLORS['text_dim'], font=FONTS['small'])

    def _run_processing(self):
        filepath = filedialog.askopenfilename(title="Select ECG Record (.dat)", filetypes=[("WFDB Data files", "*.dat")])
        if not filepath:
            return
        short = os.path.basename(filepath)
        self.filepath_label.config(text=f"  {short}", fg=COLORS['text_main'])
        self._set_status("PROCESSING…", COLORS['warning'])
        self._reset_result()
        self.progress.pack(fill='x', pady=(8, 0))
        self.progress.start(12)
        threading.Thread(target=self._process, args=(filepath,), daemon=True).start()

    def _process(self, filepath):
        record_path = filepath.rsplit('.', 1)[0]
        try:
            record = wfdb.rdrecord(record_path)
            signal = apply_bandpass_filter(record.p_signal[:, 0])
            beats, r_locs = extract_heartbeats(signal)
            if not beats:
                self.root.after(0, lambda: self._show_error("No heartbeats detected in record."))
                return
                
            features = get_combined_features(beats,r_locs, best_wavelet)
            features_scaled = best_scaler.transform(features)

            raw_preds = best_model.predict(features_scaled)

            vote_counts = Counter(raw_preds)
            total_beats = len(raw_preds)

            most_common_id, count = vote_counts.most_common(1)[0]
            confidence = count / total_beats

            if confidence <= 0.80:
                most_common_id = None
                # confidence = 0.0

            self.root.after(0, lambda: self._display_result(most_common_id, confidence))
        except Exception as e:
            err_msg = str(e)[:60]
            self.root.after(0, lambda: self._show_error(f"Read error: {err_msg}"))

    def _display_result(self, person_id, confidence):
        self.progress.stop()
        self.progress.pack_forget()
        if confidence > 0.80:
            color = COLORS['success']
            label_text = f"SUBJECT  {person_id}  IDENTIFIED"
            id_text = f"S{person_id:02d}"
            photo_file = f"person_{person_id}.jpg"
            self._set_status("MATCH  FOUND", COLORS['success'])
        else:
            color = COLORS['danger']
            label_text = "UNIDENTIFIED  SUBJECT"
            id_text = "??"
            photo_file = "unknown.jpg"
            self._set_status("NO  MATCH", COLORS['danger'])
        self.result_id_label.config(text=id_text, fg=color)
        self.result_label.config(text=label_text, fg=color)
        self.conf_label.config(text=f"{confidence * 100:.1f}%", fg=color)
        target_w = int(self._bar_total_width * confidence) if self._bar_total_width else 0
        bar_color = color
        self.conf_bar.config(bg=bar_color, width=target_w)
        self._show_photo(photo_file, person_id if confidence > 0.80 else None)

    def _show_photo(self, filename, person_id=None):
        try:
            img = Image.open(filename).convert('RGB')
            img = img.resize((300, 320), Image.Resampling.LANCZOS)
        except Exception:
            img = self._create_fallback_image(person_id)
        photo = ImageTk.PhotoImage(img)
        self._current_photo = photo
        self.photo_canvas.delete("all")
        self.photo_canvas.create_image(0, 0, anchor='nw', image=photo)
        self._draw_brackets(self.photo_canvas, 300, 320)
        name = f"SUBJECT  {person_id:02d}" if person_id else "UNKNOWN  SUBJECT"
        self.photo_name_label.config(text=name, fg=COLORS['success'] if person_id else COLORS['danger'])

    def _create_fallback_image(self, person_id):
        img = Image.new('RGB', (300, 320), color='#0A0E1A')
        draw = ImageDraw.Draw(img)
        for x in range(0, 300, 20):
            draw.line([(x, 0), (x, 320)], fill='#1A2235', width=1)
        for y in range(0, 320, 20):
            draw.line([(0, y), (300, y)], fill='#1A2235', width=1)
        draw.ellipse([110, 60, 190, 140], outline='#1E2D50', width=2)
        draw.arc([60, 130, 240, 260], start=0, end=180, fill='#1E2D50', width=2)
        label = f"P{person_id:02d}" if person_id else "??"
        draw.text((138, 275), label, fill='#6B7A99')
        return img

    def _draw_brackets(self, canvas, w, h):
        l = 16
        t = 2
        c = COLORS['accent']
        canvas.create_line(0, 0, l, 0, fill=c, width=t)
        canvas.create_line(0, 0, 0, l, fill=c, width=t)
        canvas.create_line(w - l, 0, w, 0, fill=c, width=t)
        canvas.create_line(w, 0, w, l, fill=c, width=t)
        canvas.create_line(0, h - l, 0, h, fill=c, width=t)
        canvas.create_line(0, h, l, h, fill=c, width=t)
        canvas.create_line(w - l, h, w, h, fill=c, width=t)
        canvas.create_line(w, h - l, w, h, fill=c, width=t)

    def _show_error(self, msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.result_label.config(text=f"ERROR: {msg}", fg=COLORS['danger'])
        self.result_id_label.config(text="!!", fg=COLORS['danger'])
        self._set_status("ERROR", COLORS['danger'])

    def _reset_result(self):
        self.result_id_label.config(text="—", fg=COLORS['text_dim'])
        self.result_label.config(text="Processing ECG signal…", fg=COLORS['text_dim'])
        self.conf_label.config(text="—", fg=COLORS['accent'])
        self.conf_bar.config(width=0)
        self._draw_placeholder()
        self.photo_name_label.config(text="SCANNING…", fg=COLORS['warning'])

    def _set_status(self, text, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 10, 10, fill=color, outline='')

    def _start_ecg_animation(self):
        self._animating = True
        self._animate_ecg()

    def _animate_ecg(self):
        if not self._animating:
            return
        w = self.ecg_canvas.winfo_width() or 1200
        draw_ecg_line(self.ecg_canvas, w, 60, COLORS['accent'], self._ecg_offset)
        draw_ecg_line(self.ecg_canvas, w, 60, '#003D55', (self._ecg_offset + 60) % 120)
        self._ecg_offset = (self._ecg_offset + 3) % 120
        self.root.after(40, self._animate_ecg)

if __name__ == "__main__":
    if best_model and best_scaler:
        root = tk.Tk()
        app = BiometricApp(root)
        root.mainloop()
    else:
        print("Model training failed. Please check your data paths.")
