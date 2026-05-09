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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        coeffs = pywt.wavedec(beat, wavelet_name, level=4)
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
all_results      = []

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
    print(f"  {'Classifier':<18} {'Parameters':<40} {'Accuracy':>10}")
    print(f"  {'-'*18} {'-'*40} {'-'*10}")

    wv_best = {}
    best_svm_acc = 0; best_svm = None
    for p in svm_params:
        clf = SVC(**p, probability=True)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
        param_str = ', '.join(f"{k}={v}" for k, v in p.items())
        all_results.append((wv, 'SVM', param_str, acc))
        print(f"  {'SVM':<18} {param_str:<40} {acc:>9.2f}%")
        if acc > best_svm_acc:
            best_svm_acc = acc; best_svm = clf
    wv_best['SVM'] = (best_svm, best_svm_acc)

    best_rf_acc = 0; best_rf = None
    for p in rf_params:
        clf = RandomForestClassifier(**p, random_state=42)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
        param_str = ', '.join(f"{k}={v}" for k, v in p.items())
        all_results.append((wv, 'Random Forest', param_str, acc))
        print(f"  {'Random Forest':<18} {param_str:<40} {acc:>9.2f}%")
        if acc > best_rf_acc:
            best_rf_acc = acc; best_rf = clf
    wv_best['Random Forest'] = (best_rf, best_rf_acc)

    best_knn_acc = 0; best_knn = None
    for p in knn_params:
        clf = KNeighborsClassifier(**p)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
        param_str = ', '.join(f"{k}={v}" for k, v in p.items())
        all_results.append((wv, 'KNN', param_str, acc))
        print(f"  {'KNN':<18} {param_str:<40} {acc:>9.2f}%")
        if acc > best_knn_acc:
            best_knn_acc = acc; best_knn = clf
    wv_best['KNN'] = (best_knn, best_knn_acc)

    wavelet_results[wv] = (wv_best, scaler)
    for clf_name, (clf, acc) in wv_best.items():
        if acc > best_overall_acc:
            best_overall_acc = acc; best_model = clf; best_scaler = scaler; best_wavelet = wv

# ── Summary Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"  {'Rank':<5} {'Wavelet':<8} {'Classifier':<18} {'Parameters':<35} {'Accuracy':>9}")
print(f"  {'-'*5} {'-'*8} {'-'*18} {'-'*35} {'-'*9}")
for rank, (wv, clf_name, params, acc) in enumerate(
        sorted(all_results, key=lambda x: -x[3]), start=1):
    print(f"  {rank:<5} {wv:<8} {clf_name:<18} {params:<35} {acc:>8.2f}%")

print("\n" + "=" * 75)
clf_display_names = {'SVC': 'SVM', 'RandomForestClassifier': 'Random Forest', 'KNeighborsClassifier': 'KNN'}
best_clf_name = clf_display_names.get(type(best_model).__name__, type(best_model).__name__)
print(f"  * Overall Best: {best_clf_name} | Wavelet: {best_wavelet} | Accuracy: {best_overall_acc:.2f}%\n")

# ─────────────────────────────────────────────
#  Pre-computed GUI Data
# ─────────────────────────────────────────────
DARK   = "#1C1C2E"
LIGHT  = "#F5F5F5"
BLUE   = "#1A73E8"
GREEN  = "#34A853"
RED    = "#EA4335"
ORANGE = "#FBBC05"
PURPLE = "#7B2D8B"

_cm_preds = best_model.predict(best_scaler.transform(
            get_combined_features(X_test_raw, R_test, best_wavelet)))
cm_data = confusion_matrix(y_test, _cm_preds)

results_table_data = sorted(
    [(r[0], r[1], r[2], f"{r[3]:.2f}%") for r in all_results],
    key=lambda x: -float(x[3][:-1])
)

best_per_clf = []
for _cn in ['SVM', 'Random Forest', 'KNN']:
    _top_acc, _top_wv = 0, ''
    for _wv, (_wb, _) in wavelet_results.items():
        _, _a = _wb[_cn]
        if _a > _top_acc:
            _top_acc, _top_wv = _a, _wv
    best_per_clf.append([_cn, _top_wv, f"{_top_acc:.2f}%"])

# ─────────────────────────────────────────────
#  Design System
# ─────────────────────────────────────────────

C = {
    # Backgrounds
    'bg':        '#0D1117',
    'surface':   '#161B22',
    'card':      '#1C2333',
    'raised':    '#21262D',
    # Accents
    'blue':      '#4493F8',
    'blue_dim':  '#1F6FEB',
    'cyan':      '#39C5CF',
    'green':     '#3FB950',
    'green_dim': '#238636',
    'amber':     '#D29922',
    'red':       '#F85149',
    'purple':    '#BC8CFF',
    # Text
    'text':      '#E6EDF3',
    'muted':     '#8B949E',
    'subtle':    '#484F58',
    # Borders
    'border':    '#30363D',
    'border_hi': '#6E7681',
}

# Font stack
_UI   = ('Segoe UI', 'SF Pro Display', 'Helvetica Neue', 'Helvetica', 'Arial')
_MONO = ('Consolas', 'SF Mono', 'Menlo', 'Courier New', 'Courier')

def _ui(size, weight='normal'):
    return (_UI[0], size, weight)

def _mono(size, weight='normal'):
    return (_MONO[0], size, weight)

F = {
    'heading':    _ui(18, 'bold'),
    'subheading': _ui(11),
    'label':      _ui(11, 'bold'),
    'body':       _ui(10),
    'small':      _ui(9),
    'caption':    _ui(8),
    'data_xl':    _mono(32, 'bold'),
    'data_lg':    _mono(15, 'bold'),
    'data_md':    _mono(11, 'bold'),
    'data_sm':    _mono(9),
}

def draw_ecg_line(canvas, width, height, color, offset=0, lw=2):
    canvas.delete("ecg")
    pts = []
    x = 0
    step = 3
    while x < width:
        phase = (x + offset) % 140
        if   phase < 42:  y = height // 2
        elif phase < 47:  y = height // 2 + 5
        elif phase < 52:  y = height // 2 - 32
        elif phase < 57:  y = height // 2 + 10
        elif phase < 62:  y = height // 2 - 4
        elif phase < 85:  y = height // 2 - int(7 * math.sin(math.pi * (phase - 62) / 23))
        else:             y = height // 2
        pts.extend([x, y])
        x += step
    if len(pts) >= 4:
        canvas.create_line(pts, fill=color, width=lw, smooth=True, tags="ecg")

# ─────────────────────────────────────────────
#  Widget Helpers
# ─────────────────────────────────────────────

def _sep(parent, orient='h', pad=0, color=None):
    color = color or C['border']
    if orient == 'h':
        tk.Frame(parent, bg=color, height=1).pack(fill='x', padx=pad)
    else:
        tk.Frame(parent, bg=color, width=1).pack(fill='y', pady=pad)

def _label(parent, text, font=None, fg=None, bg=None, **kw):
    return tk.Label(parent,
                    text=text,
                    font=font or F['body'],
                    fg=fg or C['text'],
                    bg=bg or C['card'],
                    **kw)

def _pill(parent, text, color, bg=None):
    """Small colored badge."""
    bg = bg or C['card']
    f = tk.Frame(parent, bg=color, padx=0, pady=0)
    tk.Label(f, text=f"  {text}  ", font=F['caption'],
             bg=color, fg='#FFFFFF', pady=2).pack()
    return f


# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────

class BiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Biometric Authentication System")
        self.root.state('zoomed')
        self.root.configure(bg=C['bg'])
        self._ecg_offset   = 0
        self._animating    = False
        self._current_photo = None
        self._prep         = None
        self._btn_hover    = False
        self._build_ui()
        self._start_ecg_animation()

    # ──────────────────────────────────────────────────────────────
    #  Root Layout
    # ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_sidebar()
        self._build_main_area()
        self._build_statusbar()

    # ──────────────────────────────────────────────────────────────
    #  Left Sidebar
    # ──────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=C['surface'], width=220)
        sb.pack(side='left', fill='y')
        sb.pack_propagate(False)

        logo_frame = tk.Frame(sb, bg=C['surface'], pady=20, padx=18)
        logo_frame.pack(fill='x')

        icon_row = tk.Frame(logo_frame, bg=C['surface'])
        icon_row.pack(anchor='w')
        icon_bg = tk.Frame(icon_row, bg=C['blue'], width=36, height=36)
        icon_bg.pack(side='left')
        icon_bg.pack_propagate(False)
        tk.Label(icon_bg, text='H', font=_ui(16, 'bold'),
                 bg=C['blue'], fg='white').pack(expand=True)
        title_col = tk.Frame(icon_row, bg=C['surface'], padx=10)
        title_col.pack(side='left')
        tk.Label(title_col, text='ECG-Lock', font=_ui(13, 'bold'),
                 bg=C['surface'], fg=C['text']).pack(anchor='w')
        tk.Label(title_col, text='v2.0', font=F['caption'],
                 bg=C['surface'], fg=C['muted']).pack(anchor='w')

        _sep(sb, pad=18)

        ecg_wrap = tk.Frame(sb, bg=C['surface'], padx=18, pady=10)
        ecg_wrap.pack(fill='x')
        tk.Label(ecg_wrap, text='LIVE SIGNAL', font=F['caption'],
                 bg=C['surface'], fg=C['muted']).pack(anchor='w', pady=(0,4))
        self.ecg_canvas = tk.Canvas(ecg_wrap, height=42, bg=C['bg'],
                                    highlightthickness=1,
                                    highlightbackground=C['border'])
        self.ecg_canvas.pack(fill='x')

        _sep(sb, pad=18)

        nav_frame = tk.Frame(sb, bg=C['surface'], pady=6)
        nav_frame.pack(fill='x')
        tk.Label(nav_frame, text='NAVIGATION', font=F['caption'],
                 bg=C['surface'], fg=C['muted'], padx=18).pack(anchor='w', pady=(0,6))

        self._nav_buttons = []
        nav_items = [
            ('>', 'Authentication', 0),
            ('#', 'Preprocessing',  1),
            ('*', 'Classifier Results', 2),
        ]
        self._selected_tab = tk.IntVar(value=0)
        self._nb_ref = None

        for icon, label, idx in nav_items:
            btn = tk.Frame(sb, bg=C['surface'], cursor='hand2')
            btn.pack(fill='x', padx=12, pady=2)
            inner = tk.Frame(btn, bg=C['surface'], padx=6, pady=8)
            inner.pack(fill='x')
            tk.Label(inner, text=icon, font=_ui(12),
                     bg=C['surface'], fg=C['text']).pack(side='left', padx=(4,8))
            lbl = tk.Label(inner, text=label, font=F['body'],
                           bg=C['surface'], fg=C['muted'], anchor='w')
            lbl.pack(side='left', fill='x', expand=True)
            indicator = tk.Frame(btn, bg=C['surface'], width=3, height=36)
            indicator.place(relx=1.0, rely=0.5, anchor='e', x=-1)

            def _on_click(i=idx, b=btn, l=lbl, ind=indicator):
                self._select_nav(i, b, l, ind)
            btn.bind('<Button-1>', lambda e, fn=_on_click: fn())
            inner.bind('<Button-1>', lambda e, fn=_on_click: fn())
            lbl.bind('<Button-1>', lambda e, fn=_on_click: fn())
            self._nav_buttons.append((btn, lbl, indicator))

        self._select_nav(0, *self._nav_buttons[0])

        tk.Frame(sb, bg=C['surface']).pack(fill='both', expand=True)

        _sep(sb, pad=0)
        info_frame = tk.Frame(sb, bg=C['surface'], padx=18, pady=14)
        info_frame.pack(fill='x')
        for k, v in [('Model', best_clf_name), ('Wavelet', best_wavelet),
                     ('Accuracy', f'{best_overall_acc:.1f}%')]:
            row = tk.Frame(info_frame, bg=C['surface'])
            row.pack(fill='x', pady=1)
            tk.Label(row, text=k, font=F['caption'],
                     bg=C['surface'], fg=C['muted'], width=9, anchor='w').pack(side='left')
            tk.Label(row, text=v, font=F['data_sm'],
                     bg=C['surface'], fg=C['cyan']).pack(side='left')

    def _select_nav(self, idx, btn, lbl, indicator):
        for b, l, ind in self._nav_buttons:
            b.config(bg=C['surface'])
            for child in b.winfo_children():
                child.config(bg=C['surface'])
                for gc in child.winfo_children():
                    try: gc.config(bg=C['surface'], fg=C['muted'])
                    except: pass
            ind.config(bg=C['surface'])
        btn.config(bg=C['raised'])
        for child in btn.winfo_children():
            child.config(bg=C['raised'])
            for gc in child.winfo_children():
                try: gc.config(bg=C['raised'])
                except: pass
        lbl.config(fg=C['text'], bg=C['raised'])
        indicator.config(bg=C['blue'])
        if self._nb_ref:
            self._nb_ref.select(idx)

    # ──────────────────────────────────────────────────────────────
    #  Main Content Area
    # ──────────────────────────────────────────────────────────────
    def _build_main_area(self):
        main = tk.Frame(self.root, bg=C['bg'])
        main.pack(side='left', fill='both', expand=True)

        header = tk.Frame(main, bg=C['surface'], height=56)
        header.pack(fill='x')
        header.pack_propagate(False)

        hdr_inner = tk.Frame(header, bg=C['surface'], padx=24)
        hdr_inner.pack(fill='both', expand=True)

        self.page_title = tk.Label(hdr_inner, text='Authentication',
                                   font=F['heading'], bg=C['surface'], fg=C['text'])
        self.page_title.pack(side='left', pady=12)

        status_area = tk.Frame(hdr_inner, bg=C['surface'])
        status_area.pack(side='right', pady=12)

        self._status_frame = tk.Frame(status_area, bg=C['green_dim'],
                                      padx=10, pady=4)
        self._status_frame.pack()
        self._status_dot = tk.Canvas(self._status_frame, width=8, height=8,
                                     bg=C['green_dim'], highlightthickness=0)
        self._status_dot.pack(side='left', padx=(0,5))
        self._status_dot.create_oval(1, 1, 7, 7, fill=C['green'], outline='')
        self._status_text = tk.Label(self._status_frame, text='SYSTEM READY',
                                     font=F['data_sm'], bg=C['green_dim'], fg=C['green'])
        self._status_text.pack(side='left')

        _sep(main)

        sty = ttk.Style()
        sty.theme_use('default')
        sty.configure('Hidden.TNotebook', background=C['bg'], borderwidth=0, tabmargins=[0,0,0,0])
        sty.configure('Hidden.TNotebook.Tab', padding=[0,0], font=('Helvetica', 1))
        sty.layout('Hidden.TNotebook.Tab', [])

        sty.configure('Sub.TNotebook', background=C['bg'], borderwidth=0)
        sty.configure('Sub.TNotebook.Tab',
                      background=C['card'], foreground=C['muted'],
                      padding=[16, 7], font=(_UI[0], 9), borderwidth=0)
        sty.map('Sub.TNotebook.Tab',
                background=[('selected', C['raised'])],
                foreground=[('selected', C['text'])])

        sty.configure('ECG.Horizontal.TProgressbar',
                      troughcolor=C['raised'], background=C['blue'], thickness=3)

        nb = ttk.Notebook(main, style='Hidden.TNotebook')
        nb.pack(fill='both', expand=True)
        self._nb_ref = nb

        tab_auth = tk.Frame(nb, bg=C['bg'])
        tab_prep = tk.Frame(nb, bg=C['bg'])
        tab_clf  = tk.Frame(nb, bg=C['bg'])
        nb.add(tab_auth)
        nb.add(tab_prep)
        nb.add(tab_clf)

        self._build_auth_tab(tab_auth)
        self._build_prep_tab(tab_prep)
        self._build_clf_tab(tab_clf)

    # ──────────────────────────────────────────────────────────────
    #  Status bar
    # ──────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=C['surface'], height=28)
        bar.pack(side='bottom', fill='x')
        bar.pack_propagate(False)
        _sep(bar)
        inner = tk.Frame(bar, bg=C['surface'])
        inner.pack(fill='both', expand=True, padx=16)
        tk.Label(inner, text='ECG-Lock  |  db4 Wavelet + Pan-Tompkins + Fiducials  |  RBF-SVM',
                 font=F['caption'], bg=C['surface'], fg=C['subtle']).pack(side='left', pady=6)
        tk.Label(inner, text=f'Best Accuracy: {best_overall_acc:.2f}%',
                 font=F['caption'], bg=C['surface'], fg=C['subtle']).pack(side='right', pady=6)

    # ──────────────────────────────────────────────────────────────
    #  Card helper
    # ──────────────────────────────────────────────────────────────
    def _card(self, parent, title=None, subtitle=None, padx=0, pady=0):
        outer = tk.Frame(parent, bg=C['card'],
                         highlightbackground=C['border'], highlightthickness=1)
        if title:
            header = tk.Frame(outer, bg=C['card'], padx=16, pady=10)
            header.pack(fill='x')
            accent = tk.Frame(header, bg=C['blue'], width=3)
            accent.pack(side='left', fill='y', padx=(0, 10))
            th = tk.Frame(header, bg=C['card'])
            th.pack(side='left', fill='y')
            tk.Label(th, text=title, font=F['label'],
                     bg=C['card'], fg=C['text']).pack(anchor='w')
            if subtitle:
                tk.Label(th, text=subtitle, font=F['caption'],
                         bg=C['card'], fg=C['muted']).pack(anchor='w')
            _sep(outer, color=C['border'])
        body = tk.Frame(outer, bg=C['card'], padx=16+padx, pady=14+pady)
        body.pack(fill='both', expand=True)
        return outer, body

    # ──────────────────────────────────────────────────────────────
    #  Tab 1 – Authentication
    # ──────────────────────────────────────────────────────────────
    def _build_auth_tab(self, parent):
        wrapper = tk.Frame(parent, bg=C['bg'])
        wrapper.pack(fill='both', expand=True, padx=24, pady=20)

        left = tk.Frame(wrapper, bg=C['bg'])
        left.pack(side='left', fill='both', expand=True)

        right = tk.Frame(wrapper, bg=C['bg'], width=300)
        right.pack(side='right', fill='y', padx=(20, 0))
        right.pack_propagate(False)

        c1, b1 = self._card(left, 'Load ECG Record',
                             'Select a WFDB .dat file to begin authentication')
        c1.pack(fill='x', pady=(0, 14))
        self._build_load_section(b1)

        c2, b2 = self._card(left, 'Analysis Configuration',
                             'Active processing pipeline')
        c2.pack(fill='x', pady=(0, 14))
        self._build_params_section(b2)

        c3, b3 = self._card(left, 'Identification Result',
                             'Subject authentication outcome')
        c3.pack(fill='x', pady=(0, 14))
        self._build_result_section(b3)

        c4, b4 = self._card(right, 'Subject Profile', padx=-8, pady=-6)
        c4.pack(fill='both', expand=True)
        self._build_photo_section(b4)

    def _build_load_section(self, parent):
        desc = tk.Label(parent,
                        text='Load a WFDB record (.dat) to extract heartbeats, compute\n'
                             'wavelet features, and identify the subject.',
                        font=F['body'], bg=C['card'], fg=C['muted'],
                        justify='left', anchor='w')
        desc.pack(anchor='w', pady=(0, 12))

        btn_frame = tk.Frame(parent, bg=C['card'])
        btn_frame.pack(anchor='w')
        self.load_btn = tk.Button(btn_frame,
                                  text='  >   Load ECG Record',
                                  command=self._run_processing,
                                  font=F['label'],
                                  bg=C['blue'], fg='white',
                                  activebackground=C['blue_dim'],
                                  activeforeground='white',
                                  relief='flat', cursor='hand2',
                                  padx=22, pady=9, bd=0)
        self.load_btn.pack(side='left')

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(parent, variable=self.progress_var,
                                        maximum=100, mode='indeterminate',
                                        style='ECG.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=(10, 0))
        self.progress.pack_forget()

        self.filepath_label = tk.Label(parent, text='No file selected',
                                       font=F['data_sm'], bg=C['card'],
                                       fg=C['subtle'], anchor='w')
        self.filepath_label.pack(fill='x', pady=(8, 0))

    def _build_params_section(self, parent):
        rows = [
            ('Wavelet',        'Daubechies db4',               C['cyan']),
            ('Fiducials',      'Pan-Tompkins + Wing Function', C['text']),
            ('Classifier',     best_clf_name,                  C['blue']),
            ('Features',       'Wavelet Energy + Intervals',   C['text']),
            ('Vote Threshold', '> 80% beat consensus',         C['amber']),
        ]
        grid = tk.Frame(parent, bg=C['card'])
        grid.pack(fill='x')
        for i, (key, val, vc) in enumerate(rows):
            bg = C['raised'] if i % 2 == 0 else C['card']
            row = tk.Frame(grid, bg=bg, pady=6, padx=10)
            row.pack(fill='x')
            tk.Label(row, text=key, font=F['small'],
                     bg=bg, fg=C['muted'], width=15, anchor='w').pack(side='left')
            tk.Label(row, text=val, font=F['data_sm'],
                     bg=bg, fg=vc, anchor='w').pack(side='left')

    def _build_result_section(self, parent):
        top = tk.Frame(parent, bg=C['card'])
        top.pack(fill='x', pady=(0, 12))

        self.result_id_label = tk.Label(top, text='--',
                                        font=F['data_xl'],
                                        bg=C['card'], fg=C['subtle'])
        self.result_id_label.pack(side='left', padx=(0, 20))

        right_col = tk.Frame(top, bg=C['card'])
        right_col.pack(side='left', fill='y')
        self.result_label = tk.Label(right_col,
                                     text='Awaiting ECG input...',
                                     font=F['data_md'],
                                     bg=C['card'], fg=C['subtle'])
        self.result_label.pack(anchor='w')
        self.result_sub = tk.Label(right_col, text='Load a record to begin',
                                   font=F['small'], bg=C['card'], fg=C['subtle'])
        self.result_sub.pack(anchor='w', pady=(3, 0))

        _sep(parent, color=C['border'])

        conf_row = tk.Frame(parent, bg=C['card'], pady=8)
        conf_row.pack(fill='x')
        tk.Label(conf_row, text='CONFIDENCE', font=F['caption'],
                 bg=C['card'], fg=C['muted']).pack(side='left')
        self.conf_label = tk.Label(conf_row, text='-',
                                   font=F['data_md'],
                                   bg=C['card'], fg=C['subtle'])
        self.conf_label.pack(side='right')

        bar_track = tk.Frame(parent, bg=C['raised'], height=5)
        bar_track.pack(fill='x')
        bar_track.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_track, bg=C['subtle'], height=5, width=0)
        self.conf_bar.place(x=0, y=0, height=5)
        self._bar_total_width = 0
        bar_track.bind('<Configure>',
                       lambda e: setattr(self, '_bar_total_width', e.width))

    def _build_photo_section(self, parent):
        self.photo_canvas = tk.Canvas(parent, width=268, height=290,
                                      bg=C['bg'],
                                      highlightbackground=C['border'],
                                      highlightthickness=1)
        self.photo_canvas.pack(fill='x')
        self._draw_placeholder()

        self.photo_name_label = tk.Label(parent, text='UNKNOWN',
                                         font=F['data_md'],
                                         bg=C['card'], fg=C['subtle'])
        self.photo_name_label.pack(pady=(10, 4))

        self.photo_sub_label = tk.Label(parent, text='No subject identified',
                                        font=F['small'],
                                        bg=C['card'], fg=C['subtle'])
        self.photo_sub_label.pack()

    def _draw_placeholder(self):
        self.photo_canvas.delete("all")
        c = self.photo_canvas
        w, h = 268, 290
        for i in range(0, w, 24):
            c.create_line(i, 0, i, h, fill='#1C2333', width=1)
        for i in range(0, h, 24):
            c.create_line(0, i, w, i, fill='#1C2333', width=1)
        c.create_oval(94, 50, 174, 130, outline=C['border'], width=2)
        c.create_arc(44, 125, 224, 248, start=0, extent=180,
                     outline=C['border'], width=2, style='arc')
        c.create_text(134, 265, text='NO SUBJECT LOADED',
                      fill=C['subtle'], font=F['caption'])
        self._draw_brackets(c, w, h, C['subtle'])

    # ──────────────────────────────────────────────────────────────
    #  Tab 2 – Preprocessing
    # ──────────────────────────────────────────────────────────────
    def _build_prep_tab(self, parent):
        self._load_prep_data()

        header = tk.Frame(parent, bg=C['bg'], padx=24, pady=14)
        header.pack(fill='x')
        tk.Label(header, text='Signal Preprocessing Pipeline',
                 font=F['heading'], bg=C['bg'], fg=C['text']).pack(side='left')

        _sep(parent)

        sub_nb = ttk.Notebook(parent, style='Sub.TNotebook')
        sub_nb.pack(fill='both', expand=True, padx=14, pady=14)

        specs = [
            ('  Step 1 & 2  -  Filter  ',        self._fig_raw_filtered),
            ('  Step 3 & 4  -  Threshold  ',      self._fig_diff_threshold),
            ('  Step 5 & 6  -  R-peaks & Beat  ', self._fig_rpeaks_beat),
            ('  Fiducial Points  ',                self._fig_fiducial),
            ('  Wavelet  db4  ',                   self._fig_wavelet),
        ]
        for title, fig_fn in specs:
            frame = tk.Frame(sub_nb, bg=C['bg'])
            sub_nb.add(frame, text=title)
            try:
                fig = fig_fn()
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
            except Exception as e:
                tk.Label(frame, text=f'Could not render plot:\n{e}',
                         bg=C['bg'], fg=C['red'], font=F['body']).pack(expand=True)

    def _load_prep_data(self):
        try:
            rec  = wfdb.rdrecord(os.path.join(base_data_path, 'Person_1', 'rec_1'))
            fs   = rec.fs
            raw  = rec.p_signal[:fs * 5, 0]
            filt = apply_bandpass_filter(raw, fs)
            diff_sig = np.zeros_like(filt)
            diff_sig[1:] = np.diff(filt)
            sq_sig = diff_sig ** 2
            wl     = int(0.15 * fs)
            integ  = np.convolve(sq_sig, np.ones(wl) / wl, mode='same')
            thresh = np.mean(integ) + 1.5 * np.std(integ)
            r_peaks = pan_tompkins_r_peaks(filt, fs)
            before, after = int(0.2 * fs), int(0.4 * fs)
            beat = None
            for r in r_peaks:
                if r - before >= 0 and r + after < len(filt):
                    beat = filt[r - before: r + after]; break
            beat_fid, pts = None, None
            for r in r_peaks:
                if r - before >= 0 and r + after < len(filt):
                    b = filt[r - before: r + after]
                    p = extract_fiducial_points(b, before, fs)
                    if all(v is not None for v in p.values()):
                        beat_fid, pts = b, p; break
            self._prep = dict(fs=fs, raw=raw, filt=filt, r_peaks=r_peaks,
                              diff_sig=diff_sig, integ=integ, thresh=thresh,
                              beat=beat, beat_fid=beat_fid, pts=pts, before=before)
        except Exception as e:
            print(f'Prep data load error: {e}')
            self._prep = None

    # ── Preprocessing figures ─────────────────────────────────────
    def _fig_raw_filtered(self):
        d = self._prep; fs = d['fs']
        t = np.arange(len(d['raw'])) / fs
        fig = Figure(figsize=(12, 6), facecolor='white')
        fig.suptitle("Preprocessing - Step 1 & 2", fontsize=14, fontweight='bold')
        ax1 = fig.add_subplot(211)
        ax1.plot(t, d['raw'], color='#888888', lw=0.9)
        ax1.set_title("Step 1: Raw ECG Signal")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude (mV)")
        ax1.set_facecolor(LIGHT); ax1.grid(True, alpha=0.4)
        ax2 = fig.add_subplot(212)
        ax2.plot(t, d['filt'], color=BLUE, lw=0.9)
        ax2.set_title("Step 2: After Bandpass Filter (1-40 Hz, 4th-order Butterworth)")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Amplitude (mV)")
        ax2.set_facecolor(LIGHT); ax2.grid(True, alpha=0.4)
        fig.tight_layout(); return fig

    def _fig_diff_threshold(self):
        d = self._prep; fs = d['fs']
        t = np.arange(len(d['filt'])) / fs
        from scipy.signal import find_peaks as _fp
        fig = Figure(figsize=(12, 6), facecolor='white')
        fig.suptitle("Preprocessing - Step 3 & 4", fontsize=14, fontweight='bold')
        ax1 = fig.add_subplot(211)
        ax1.plot(t, d['diff_sig'], color=ORANGE, lw=0.8)
        ax1.set_title("Step 3: Differentiation  (highlights rapid slope changes / QRS)")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("dV/dt")
        ax1.set_facecolor(LIGHT); ax1.grid(True, alpha=0.4)
        ax2 = fig.add_subplot(212)
        ax2.plot(t, d['integ'], color=ORANGE, lw=1.0, label='Energy Envelope')
        ax2.axhline(d['thresh'], color=RED, lw=1.4, ls='--',
                    label='Adaptive Threshold  (mean + 1.5xstd)')
        all_p, _ = _fp(d['integ'], distance=int(0.3 * fs))
        rej_p = [p for p in all_p if d['integ'][p] < d['thresh']]
        ax2.scatter(d['r_peaks'] / fs, d['integ'][d['r_peaks']],
                    color=GREEN, zorder=5, s=60, label='Accepted Peaks')
        ax2.scatter([p / fs for p in rej_p], d['integ'][rej_p],
                    color='black', marker='x', s=60, label='Rejected Peaks')
        ax2.set_title("Step 4: Squaring + Moving-Average Integration & Adaptive Thresholding")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Energy")
        ax2.set_facecolor(LIGHT); ax2.grid(True, alpha=0.4); ax2.legend(fontsize=9)
        fig.tight_layout(); return fig

    def _fig_rpeaks_beat(self):
        d = self._prep; fs = d['fs']
        t = np.arange(len(d['filt'])) / fs
        fig = Figure(figsize=(12, 6), facecolor='white')
        fig.suptitle("Preprocessing - Step 5 & 6", fontsize=14, fontweight='bold')
        ax1 = fig.add_subplot(211)
        ax1.plot(t, d['filt'], color=BLUE, lw=0.8, alpha=0.8, label='Filtered ECG')
        ax1.scatter(d['r_peaks'] / fs, d['filt'][d['r_peaks']],
                    color=RED, zorder=5, s=70, label='Detected R-peaks')
        ax1.set_title("Step 5: R-Peak Detection (Pan-Tompkins Algorithm)")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude (mV)")
        ax1.set_facecolor(LIGHT); ax1.grid(True, alpha=0.4); ax1.legend(fontsize=9)
        ax2 = fig.add_subplot(212)
        if d['beat'] is not None:
            beat_t = (np.arange(len(d['beat'])) - d['before']) / fs * 1000
            ax2.plot(beat_t, d['beat'], color=GREEN, lw=1.8)
            ax2.axvline(0, color=RED, lw=1.2, ls='--', label='R-peak  (t = 0 ms)')
            ax2.axvspan(-200, 0,   alpha=0.08, color=BLUE,   label='Pre-R (-200 ms)')
            ax2.axvspan(0,    400, alpha=0.08, color=ORANGE, label='Post-R (+400 ms)')
            ax2.set_title("Step 6: Single Heartbeat Segmentation (-200 ms to +400 ms)")
            ax2.set_xlabel("Time relative to R-peak (ms)"); ax2.set_ylabel("Amplitude (mV)")
            ax2.legend(fontsize=9)
        ax2.set_facecolor(LIGHT); ax2.grid(True, alpha=0.4)
        fig.tight_layout(); return fig

    def _fig_fiducial(self):
        d = self._prep; fs = d['fs']
        fig = Figure(figsize=(11, 5), facecolor='white')
        ax  = fig.add_subplot(111)
        if d['beat_fid'] is not None and d['pts'] is not None:
            beat_t = (np.arange(len(d['beat_fid'])) - d['before']) / fs * 1000
            ax.plot(beat_t, d['beat_fid'], color=BLUE, lw=2, label='ECG Beat')
            pt_clr = {'P': ORANGE, 'Q': GREEN, 'R': RED, 'S': PURPLE, 'T': '#0066CC'}
            pt_lbl = {'P': 'P-wave peak', 'Q': 'Q-point (QRS onset)',
                      'R': 'R-peak', 'S': 'S-point (QRS offset)', 'T': 'T-wave peak'}
            for name, idx in d['pts'].items():
                if idx is not None and 0 <= idx < len(d['beat_fid']):
                    x = beat_t[idx]; y = d['beat_fid'][idx]
                    ax.scatter(x, y, color=pt_clr[name], zorder=6, s=110)
                    ax.annotate(pt_lbl[name], xy=(x, y),
                                xytext=(x + 18, y + 0.045), fontsize=9,
                                color=pt_clr[name],
                                arrowprops=dict(arrowstyle='->', color=pt_clr[name], lw=1.2))
            pts = d['pts']
            if pts['P'] and pts['R']:
                px = beat_t[pts['P']]; rx = beat_t[pts['R']]
                ax.annotate('', xy=(rx, -0.13), xytext=(px, -0.13),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                ax.text((px + rx) / 2, -0.155, 'PR Interval', ha='center', fontsize=8.5)
            if pts['Q'] and pts['S']:
                qx = beat_t[pts['Q']]; sx = beat_t[pts['S']]
                ax.annotate('', xy=(sx, -0.19), xytext=(qx, -0.19),
                            arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5))
                ax.text((qx + sx) / 2, -0.215, 'QRS Duration', ha='center', fontsize=8.5, color=GREEN)
        ax.set_title("Feature Extraction - Fiducial Points (P, Q, R, S, T)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Time relative to R-peak (ms)"); ax.set_ylabel("Amplitude (mV)")
        ax.set_facecolor(LIGHT); ax.grid(True, alpha=0.4); ax.legend(fontsize=9)
        fig.tight_layout(); return fig

    def _fig_wavelet(self):
        d = self._prep
        fig = Figure(figsize=(12, 11), facecolor='white')
        fig.suptitle("Feature Extraction - Wavelet Decomposition  (db4, Level 4)\n"
                     "Features used: Mean - Std - Energy  of  cA4, cD4, cD3  =>  9 values",
                     fontsize=12, fontweight='bold')
        if d['beat'] is not None:
            coeffs = pywt.wavedec(d['beat'], 'db4', level=4)
            titles = ['cA4 - Approximation (0-15 Hz)   [USED]',
                      'cD4 - Detail Level 4 (15-30 Hz)  [USED]',
                      'cD3 - Detail Level 3 (30-60 Hz)  [USED]',
                      'cD2 - Detail Level 2 (60-125 Hz)',
                      'cD1 - Detail Level 1 (125-250 Hz)']
            clrs = [BLUE, GREEN, ORANGE, '#AAAAAA', '#AAAAAA']
            used = [True, True, True, False, False]
            for i, (c, ax) in enumerate(
                    zip(coeffs, [fig.add_subplot(5, 1, j + 1) for j in range(5)])):
                ax.plot(c, color=clrs[i], lw=1.3)
                ax.set_title(titles[i], fontsize=10)
                ax.set_ylabel("Amp.")
                ax.set_facecolor('#F0FFF4' if used[i] else LIGHT)
                ax.grid(True, alpha=0.4)
                stats = (f"Mean={np.mean(c):.4f}   Std={np.std(c):.4f}   "
                         f"Energy={np.sum(c**2):.4f}")
                ax.text(0.02, 0.78, stats, transform=ax.transAxes, fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='#AAAAAA', boxstyle='round,pad=0.3'))
        fig.tight_layout(); return fig

    # ──────────────────────────────────────────────────────────────
    #  Tab 3 – Classifier Results
    # ──────────────────────────────────────────────────────────────
    def _build_clf_tab(self, parent):
        header = tk.Frame(parent, bg=C['bg'], padx=24, pady=14)
        header.pack(fill='x')
        tk.Label(header, text='Classifier Performance Results',
                 font=F['heading'], bg=C['bg'], fg=C['text']).pack(side='left')

        pills = tk.Frame(header, bg=C['bg'])
        pills.pack(side='right')
        for txt, clr in [(f'Best: {best_clf_name}', C['blue']),
                         (f'Wavelet: {best_wavelet}', C['cyan']),
                         (f'{best_overall_acc:.1f}%', C['green'])]:
            _pill(pills, txt, clr).pack(side='left', padx=4)

        _sep(parent)

        sub_nb = ttk.Notebook(parent, style='Sub.TNotebook')
        sub_nb.pack(fill='both', expand=True, padx=14, pady=14)

        specs = [
            ('  All Results  ',         self._fig_all_results),
            ('  Best per Classifier  ', self._fig_best_summary),
            ('  Accuracy Chart  ',      self._fig_accuracy_bar),
            ('  Confusion Matrix  ',    self._fig_confusion),
        ]
        for title, fig_fn in specs:
            frame = tk.Frame(sub_nb, bg=C['bg'])
            sub_nb.add(frame, text=title)
            try:
                fig = fig_fn()
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
            except Exception as e:
                tk.Label(frame, text=f'Could not render plot:\n{e}',
                         bg=C['bg'], fg=C['red'], font=F['body']).pack(expand=True)

    # ── Classifier figures ────────────────────────────────────────
    def _fig_all_results(self):
        fig = Figure(figsize=(13, 9), facecolor='white')
        ax  = fig.add_subplot(111)
        ax.axis('off')
        cols = ['Wavelet', 'Classifier', 'Parameters', 'Accuracy (%)']
        tbl  = ax.table(cellText=results_table_data, colLabels=cols,
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.95)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#CCCCCC')
            if r == 0:
                cell.set_facecolor(DARK)
                cell.set_text_props(color='white', fontweight='bold')
            else:
                v = float(results_table_data[r - 1][3][:-1])
                cell.set_facecolor('#C8F7C5' if v >= 95 else
                                   '#FFF9C4' if v >= 90 else '#FFE0E0')
        # FIX: replaced emoji with ASCII text to avoid font warnings
        ax.set_title("Classification Results - All Parameter Combinations  "
                     "(sorted by Accuracy)\n"
                     "[GREEN] >= 95%   [YELLOW] >= 90%   [RED] < 90%",
                     fontsize=12, fontweight='bold', pad=18)
        fig.tight_layout(); return fig

    def _fig_best_summary(self):
        fig = Figure(figsize=(9, 3.5), facecolor='white')
        ax  = fig.add_subplot(111)
        ax.axis('off')
        cols = ['Classifier', 'Best Wavelet', 'Best Accuracy (%)']
        tbl  = ax.table(cellText=best_per_clf, colLabels=cols,
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(14); tbl.scale(1, 3.8)
        clf_fc = {'SVM': BLUE + '33', 'Random Forest': GREEN + '33', 'KNN': ORANGE + '33'}
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#CCCCCC')
            if r == 0:
                cell.set_facecolor(DARK)
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor(clf_fc.get(best_per_clf[r - 1][0], 'white'))
        ax.set_title("Best Result per Classifier (across all wavelets)",
                     fontsize=13, fontweight='bold', pad=20)
        fig.tight_layout(); return fig

    def _fig_accuracy_bar(self):
        _wvs  = ['db1', 'db2', 'db4']
        _clfs = ['SVM', 'Random Forest', 'KNN']
        x     = np.arange(len(_clfs)); w = 0.25
        _clrs = [BLUE, GREEN, ORANGE]
        fig = Figure(figsize=(11, 5), facecolor='white')
        ax  = fig.add_subplot(111)
        ax.set_facecolor(LIGHT); ax.grid(True, alpha=0.4, axis='y')
        for i, wv in enumerate(_wvs):
            wv_best, _ = wavelet_results[wv]
            accs = [wv_best[c][1] for c in _clfs]
            bars = ax.bar(x + i * w, accs, w, label=f'Wavelet: {wv}',
                          color=_clrs[i], edgecolor='white', linewidth=0.6)
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.set_xticks(x + w); ax.set_xticklabels(_clfs, fontsize=12)
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 115)
        ax.set_title("Classifier Accuracy Comparison across Wavelets", fontsize=14, fontweight='bold')
        ax.axhline(90, color=RED, lw=1.2, ls='--', alpha=0.7, label='90% reference')
        ax.legend(fontsize=10); fig.tight_layout(); return fig

    def _fig_confusion(self):
        labels = [f'P{i}' for i in range(1, 6)]
        fig = Figure(figsize=(7, 6), facecolor='white')
        ax  = fig.add_subplot(111)
        im  = ax.imshow(cm_data, cmap='Blues')
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(labels, fontsize=11); ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix - Best: {best_clf_name} | "
                     f"Wavelet: {best_wavelet} | Acc: {best_overall_acc:.2f}%",
                     fontsize=11, fontweight='bold')
        thresh = cm_data.max() / 2
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm_data[i, j]), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm_data[i, j] > thresh else 'black')
        fig.colorbar(im, ax=ax); fig.tight_layout(); return fig

    # ──────────────────────────────────────────────────────────────
    #  Processing logic
    # ──────────────────────────────────────────────────────────────
    def _run_processing(self):
        filepath = filedialog.askopenfilename(
            title='Select ECG Record (.dat)',
            filetypes=[('WFDB Data files', '*.dat')])
        if not filepath:
            return
        self.filepath_label.config(
            text=f'  {os.path.basename(filepath)}', fg=C['text'])
        self._set_status('PROCESSING...', C['amber'], C['raised'])
        self._reset_result()
        self.progress.pack(fill='x', pady=(10, 0))
        self.progress.start(12)
        threading.Thread(target=self._process, args=(filepath,), daemon=True).start()

    def _process(self, filepath):
        record_path = filepath.rsplit('.', 1)[0]
        try:
            record  = wfdb.rdrecord(record_path)
            signal  = apply_bandpass_filter(record.p_signal[:, 0])
            beats, r_locs = extract_heartbeats(signal)
            if not beats:
                self.root.after(0, lambda: self._show_error('No heartbeats detected.'))
                return
            features        = get_combined_features(beats, r_locs, best_wavelet)
            features_scaled = best_scaler.transform(features)
            raw_preds       = best_model.predict(features_scaled)
            vote_counts     = Counter(raw_preds)
            most_common_id, count = vote_counts.most_common(1)[0]
            confidence = count / len(raw_preds)
            if confidence <= 0.80:
                most_common_id = None
            self.root.after(0, lambda: self._display_result(most_common_id, confidence))
        except Exception as e:
            msg = str(e)[:60]
            self.root.after(0, lambda: self._show_error(f'Read error: {msg}'))

    def _display_result(self, person_id, confidence):
        self.progress.stop(); self.progress.pack_forget()
        if person_id and confidence > 0.80:
            color     = C['green']
            id_text   = f'S{person_id:02d}'
            res_text  = f'SUBJECT {person_id} IDENTIFIED'
            sub_text  = f'Confidence: {confidence * 100:.1f}%'
            photo_file = f'person_{person_id}.jpg'
            self._set_status('MATCH FOUND', C['green'], C['green_dim'])
        else:
            color      = C['red']
            id_text    = '??'
            res_text   = 'UNIDENTIFIED SUBJECT'
            sub_text   = f'Confidence too low: {confidence * 100:.1f}%'
            photo_file = 'unknown.jpg'
            self._set_status('NO MATCH', C['red'], '#3D1A1A')
        self.result_id_label.config(text=id_text, fg=color)
        self.result_label.config(text=res_text, fg=color)
        self.result_sub.config(text=sub_text, fg=C['muted'])
        self.conf_label.config(text=f'{confidence * 100:.1f}%', fg=color)
        target_w = int(self._bar_total_width * confidence) if self._bar_total_width else 0
        self.conf_bar.config(bg=color, width=target_w)
        self._show_photo(photo_file, person_id if (person_id and confidence > 0.80) else None)

    def _show_photo(self, filename, person_id=None):
        try:
            img = Image.open(filename).convert('RGB').resize(
                (268, 290), Image.Resampling.LANCZOS)
        except Exception:
            img = self._create_fallback_image(person_id)
        photo = ImageTk.PhotoImage(img)
        self._current_photo = photo
        self.photo_canvas.delete("all")
        self.photo_canvas.create_image(0, 0, anchor='nw', image=photo)
        bracket_clr = C['green'] if person_id else C['red']
        self._draw_brackets(self.photo_canvas, 268, 290, bracket_clr)
        name = f'SUBJECT  {person_id:02d}' if person_id else 'UNKNOWN SUBJECT'
        name_clr = C['green'] if person_id else C['red']
        self.photo_name_label.config(text=name, fg=name_clr)
        self.photo_sub_label.config(
            text='Identity verified' if person_id else 'Authentication failed',
            fg=C['green'] if person_id else C['red'])

    def _create_fallback_image(self, person_id):
        img  = Image.new('RGB', (268, 290), color=C['bg'])
        draw = ImageDraw.Draw(img)
        for x in range(0, 268, 24):
            draw.line([(x, 0), (x, 290)], fill='#1C2333', width=1)
        for y in range(0, 290, 24):
            draw.line([(0, y), (268, y)], fill='#1C2333', width=1)
        draw.ellipse([94, 50, 174, 130], outline=C['border'], width=2)
        draw.arc([44, 125, 224, 248], start=0, end=180, fill=C['border'], width=2)
        draw.text((120, 268), f'P{person_id:02d}' if person_id else '??',
                  fill=C['muted'])
        return img

    def _draw_brackets(self, canvas, w, h, color):
        L = 18; t = 2
        segs = [(0, 0, L, 0), (0, 0, 0, L),
                (w-L, 0, w, 0), (w, 0, w, L),
                (0, h-L, 0, h), (0, h, L, h),
                (w-L, h, w, h), (w, h-L, w, h)]
        for s in segs:
            canvas.create_line(*s, fill=color, width=t)

    def _show_error(self, msg):
        self.progress.stop(); self.progress.pack_forget()
        self.result_id_label.config(text='!', fg=C['red'])
        self.result_label.config(text=f'Error: {msg}', fg=C['red'])
        self.result_sub.config(text='Check file path and format', fg=C['muted'])
        self._set_status('ERROR', C['red'], '#3D1A1A')

    def _reset_result(self):
        self.result_id_label.config(text='--', fg=C['subtle'])
        self.result_label.config(text='Processing signal...', fg=C['muted'])
        self.result_sub.config(text='Extracting heartbeats and computing features...', fg=C['subtle'])
        self.conf_label.config(text='-', fg=C['subtle'])
        self.conf_bar.config(width=0)
        self._draw_placeholder()
        self.photo_name_label.config(text='SCANNING...', fg=C['amber'])
        self.photo_sub_label.config(text='Analysis in progress', fg=C['muted'])

    def _set_status(self, text, color, bg=None):
        bg = bg or C['green_dim']
        self._status_frame.config(bg=bg)
        self._status_dot.config(bg=bg)
        self._status_dot.delete("all")
        self._status_dot.create_oval(1, 1, 7, 7, fill=color, outline='')
        self._status_text.config(text=text, fg=color, bg=bg)

    # ──────────────────────────────────────────────────────────────
    #  ECG animation
    # ──────────────────────────────────────────────────────────────
    def _start_ecg_animation(self):
        self._animating = True
        self._animate_ecg()

    def _animate_ecg(self):
        if not self._animating:
            return
        w = self.ecg_canvas.winfo_width() or 184
        # FIX: replaced C['cyan'] + '40' (invalid 8-char hex) with a plain dark cyan color
        draw_ecg_line(self.ecg_canvas, w, 42, C['blue'],  self._ecg_offset, lw=2)
        draw_ecg_line(self.ecg_canvas, w, 42, '#1A5F63',
                      (self._ecg_offset + 70) % 140, lw=1)
        self._ecg_offset = (self._ecg_offset + 3) % 140
        self.root.after(40, self._animate_ecg)


if __name__ == '__main__':
    if best_model and best_scaler:
        root = tk.Tk()
        app  = BiometricApp(root)
        root.mainloop()
    else:
        print('Model training failed. Please check your data paths.')