"""
train_model_hypersensitive.py

A rearchitected training code that is hypersensitive to small differences in input.
Key steps:
1) Parse data from marker changes or sample resets (choose the approach that matches your dataset).
2) For each trial, compute band-power features (16 channels × 4 freq bands = 64 features).
3) Scale features with StandardScaler.
4) Train a high-gamma, high-C SVM, which is extremely sensitive to small variations.
5) Save the model and scaler.

WARNING: This will likely overfit. It's just for demonstration that small changes in input
produce large changes in output predictions.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch, welch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_FOLDER = "./eeg_data"
# Example: label 0..11 for your 12 items, each file containing marker-based or sample-based
# If you use marker-based segmentation, see the code snippet in the previous answer
# If you use sample-based reset detection, use that code. Just adapt the "parse_trials()" function.
FILE_LABELS = [
    ("SM001_0_3.txt", 0),
    ("SM001_1_3.txt", 1),
    ("SM001_2_3.txt", 2),
    ("SM001_3_3.txt", 3),
    ("SM001_4_3.txt", 4),
    ("SM001_5_3.txt", 5),
    ("SM001_6_3.txt", 6),
    ("SM001_7_3.txt", 7),
    ("SM001_8_3.txt", 8),
    ("SM001_9_3.txt", 9),
    ("SM001_10_3.txt", 10),
    ("SM001_11_3.txt", 11),
]

SAMPLE_RATE = 250
DESIRED_SAMPLES = 256
NUM_CHANNELS = 16
ARTIFACT_THRESHOLD = 150e-6

# Basic bandpass/notch
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0

# Frequency bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

# -----------------------------
# FILTERING AND ARTIFACT
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(order, [low, high], btype='band')
    return b,a

def apply_bandpass(sig, lowcut, highcut, fs):
    b,a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b,a,sig)

def apply_notch(sig, freq, fs, q):
    from scipy.signal import iirnotch
    nyq = fs/2.0
    f0 = freq / nyq
    b,a = iirnotch(f0,q)
    return lfilter(b,a,sig)

def artifact_removal(eeg_array, threshold):
    for ch in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch])>threshold):
            eeg_array[ch,:] = 0.0
    return eeg_array

# -----------------------------
# BAND-POWER
# -----------------------------
def compute_band_power(sig, fs, band):
    fmin,fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs>=fmin)&(freqs<=fmax))[0]
    return np.mean(psd[idx])

def extract_bandpower(trial_data, fs):
    feats = []
    for ch in range(trial_data.shape[0]):
        sig = trial_data[ch,:]
        for (lo,hi) in BANDS.values():
            bp = compute_band_power(sig, fs, (lo,hi))
            feats.append(bp)
    return np.array(feats)  # shape ~ (64,)

# -----------------------------
# TRIAL PARSING
# (Use marker-based or sample-reset approach)
# -----------------------------
def load_data_from_file_marker(filepath, label):
    """
    Example for marker-based approach. Adjust the code as in previous examples
    if your data uses the marker column for segmentation.
    """
    print(f"[INFO] Loading (marker-based) {filepath} label={label}")
    df = pd.read_csv(filepath, sep=r'\s+|,|\t', engine='python', header=None)
    arr = df.values
    # TODO: parse arr for marker changes
    # Or see the previous "marker code" snippet for a full example.
    # For now, let's just dummy return:
    Xlist = []
    # ...
    # X = np.array(Xlist)
    # y = np.array([label]*len(Xlist))
    # return X,y

    # The user must fill this in properly, referencing the prior code if needed.
    return np.zeros((0,64)), np.zeros((0,))  # placeholder

def load_data_from_file_samplereset(filepath, label):
    """
    Example for sample-number resets approach. If your data uses "sample number = 0" to indicate a new trial.
    """
    print(f"[INFO] Loading (sample-reset) {filepath} label={label}")
    df = pd.read_csv(filepath, sep=r'\s+|,|\t', engine='python', header=None)
    df = df.dropna(axis=1, how='all')
    data_np = df.values
    # columns:
    # 0 => sampleNum
    # 1..16 => EEG
    # The rest => ignore or remove
    if data_np.shape[1]<17:
        raise ValueError("Not enough columns (need sample # + 16 EEG).")

    # parse sample# resets
    # same approach as your old code that you found "incorrect." We'll do a minimal version:
    trials = []
    current = []

    for row_idx in range(data_np.shape[0]):
        row = data_np[row_idx]
        sample_num = row[0]
        eeg_values = row[1:1+NUM_CHANNELS]

        if sample_num==0 and row_idx!=0:
            # finish old trial
            if len(current)>0:
                arr = np.stack(current, axis=-1)  # (16, #samples)
                if arr.shape[1]<DESIRED_SAMPLES:
                    pad = DESIRED_SAMPLES - arr.shape[1]
                    arr = np.pad(arr, ((0,0),(0,pad)), mode='constant')
                elif arr.shape[1]>DESIRED_SAMPLES:
                    arr = arr[:,-DESIRED_SAMPLES:]
                # filter & artifact
                for ch in range(arr.shape[0]):
                    arr[ch,:] = apply_bandpass(arr[ch,:], BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
                    arr[ch,:] = apply_notch(arr[ch,:], NOTCH_FREQ, SAMPLE_RATE, NOTCH_Q)
                arr = artifact_removal(arr, ARTIFACT_THRESHOLD)

                feats = extract_bandpower(arr, SAMPLE_RATE)
                trials.append(feats)
            current=[]

        current.append(eeg_values)

    # last trial
    if len(current)>0:
        arr = np.stack(current, axis=-1)
        if arr.shape[1]<DESIRED_SAMPLES:
            pad = DESIRED_SAMPLES - arr.shape[1]
            arr = np.pad(arr, ((0,0),(0,pad)), mode='constant')
        elif arr.shape[1]>DESIRED_SAMPLES:
            arr = arr[:,-DESIRED_SAMPLES:]
        # filter & artifact
        for ch in range(arr.shape[0]):
            arr[ch,:] = apply_bandpass(arr[ch,:], BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
            arr[ch,:] = apply_notch(arr[ch,:], NOTCH_FREQ, SAMPLE_RATE, NOTCH_Q)
        arr = artifact_removal(arr, ARTIFACT_THRESHOLD)
        feats = extract_bandpower(arr, SAMPLE_RATE)
        trials.append(feats)

    X = np.array(trials)
    y = np.array([label]*len(trials))
    print(f"[INFO] Found {len(trials)} sample-reset trials in {filepath}")
    return X,y


# -----------------------------
def main():
    # choose whichever function matches your actual data segmentation logic
    # e.g. load_data_from_file_marker or load_data_from_file_samplereset
    # We'll do the sample-reset approach by default:
    load_func = load_data_from_file_samplereset
    # Or marker-based:
    # load_func = load_data_from_file_marker

    allX=[]
    allY=[]
    for fname,lbl in FILE_LABELS:
        path = os.path.join(DATA_FOLDER,fname)
        X_,y_ = load_func(path,lbl)
        allX.append(X_)
        allY.append(y_)

    X = np.concatenate(allX, axis=0)
    y = np.concatenate(allY, axis=0)
    print(f"[INFO] Final dataset shape: X={X.shape}, y={y.shape}")

    if X.shape[0]==0:
        raise ValueError("No trials loaded. Check your parse logic.")

    # scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # define a hyper-hypersensitive SVM
    # extremely large C & gamma => model changes drastically on small input changes
    model = SVC(kernel='rbf', probability=True,
                C=1e6,     # large C = less regularization
                gamma=100, # large gamma = super sensitive boundary
                class_weight=None)

    # train-test
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xs,y, test_size=0.2, random_state=42, stratify=y)
    model.fit(Xtrain, ytrain)
    from sklearn.metrics import classification_report
    ypred = model.predict(Xtest)
    print(classification_report(ytest, ypred))

    # retrain on entire data if you want
    model.fit(Xs,y)
    joblib.dump(model, "eeg_bandpower_model.joblib")
    joblib.dump(scaler,"scaler.joblib")
    print("[INFO] Hypersensitive model + scaler saved.")


if __name__=="__main__":
    main()
