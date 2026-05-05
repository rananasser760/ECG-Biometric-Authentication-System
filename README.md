# 🫀 ECG Biometric Authentication System

An ECG-based biometric authentication system that identifies individuals using their heart signals.  
This project combines signal processing, wavelet feature extraction, and machine learning, wrapped in an interactive GUI.

> 🎓 Developed as part of the **Human-Computer Interaction (HCI)** course.

---

## Features

- ECG signal preprocessing (Bandpass filtering)
- Heartbeat segmentation
- Fiducial points detection (P, Q, R, S, T)
- Wavelet-based feature extraction (db1, db2, db4)
- Machine Learning models:
  - SVM (Best performance)
  - Random Forest
  - KNN
- Interactive GUI using Tkinter
- Real-time identification with confidence score
- Subject visualization with photo display

---

## Model Overview

The system evaluates multiple combinations of wavelets and classifiers.

**Best configuration:**
- Wavelet: `db4`
- Classifier: SVM (RBF Kernel)
- Features:
  - Wavelet statistical features
  - ECG interval features (PR, QT, QRS)

---

## 📊 Dataset

- The dataset consists of ECG records stored in **WFDB format** (`.atr`, `.dat`, `.hea`)
- Data is organized per subject:
    `data/
        Person_1/
              rec_1.atr
              rec_1.dat
              rec_1.hea
              ...
        Person_2/
              rec_1.atr
              rec_1.dat
              rec_1.hea
              ...
        ...`

### Notes
- The dataset is **open-source and lightweight from Physionet**
- Included in this repository for demonstration purposes
- Each ECG signal is processed to extract individual heartbeats

---

## Project Pipeline

1. **Load ECG Signal**
 - Read signal using WFDB

2. **Preprocessing**
 - Apply bandpass filter (1–40 Hz)
 - Remove noise

3. **Heartbeat Extraction**
 - Detect R-peaks
 - Segment individual heartbeats

4. **Fiducial Point Detection**
 - Identify P, Q, R, S, T points
 - Compute ECG intervals:
   - PR interval
   - QT interval
   - QRS duration

5. **Feature Extraction**
 - Apply wavelet decomposition
 - Extract statistical features:
   - Mean
   - Standard deviation
   - Energy

6. **Feature Scaling**
 - Normalize features using StandardScaler

7. **Model Prediction**
 - Predict identity using trained classifier
 - Apply probability threshold
 - Majority voting across heartbeats

8. **Final Decision**
 - Identify subject or mark as unknown
 - Display confidence score

---

## GUI Overview

- ECG animated header
- Record upload system
- Real-time processing feedback
- Identification result with confidence bar
- Subject photo display

---

## Installation

```bash
pip install -r requirements.txt
```
##  Run the Project
- python main.py

## Project Structure
ECG-Biometric-Authentication/
│
├── data/            # ECG dataset (WFDB format)
├── images/          # Subject images
├── main.py          # Main application
├── requirements.txt
├── README.md
├── .gitignore


## Team Members:
- Rana Nasser
- Esraa Taha
- Gihad Mahmoud
- Menna Mahrous
- Mona Khaled
