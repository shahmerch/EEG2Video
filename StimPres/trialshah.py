"""
train_shah.py

End-to-end training script that:
1) Reads 12 EEG data files (each with 10 trials) from an OpenBCI Cyton, 
   where each trial is indicated by sample-number reset to 0.
2) Filters each trial (bandpass 1–50 Hz, 60 Hz notch), 
   does basic artifact removal, then computes band-power features 
   (16 channels × 4 freq bands = 64 features).
3) Trains an SVM (with probability=True) on these features, 
   so we can get a probability distribution over 12 classes.
4) Saves the final SVM model to "eeg_image_classifier.joblib" 
   and a StandardScaler to "scaler.joblib".

You'll want to place your .txt files in "eeg_data" 
(e.g. "SM001_0_3.txt", "SM001_1_3.txt", ... "SM001_11_3.txt"),
each containing 10 trials indicated by sample-number resetting to 0.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch, welch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# --------------------------
# CONFIG
# --------------------------
DATA_FOLDER = "./eeg_data"
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
NUM_CLASSES = 12
NUM_CHANNELS = 16
SAMPLE_RATE = 250
DESIRED_SAMPLES = 256  # we trim/pad each trial to 256 samples

BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0
ARTIFACT_THRESHOLD = 150e-6  # naive threshold

# Frequency bands for band-power
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30)
}


# --------------------------
# FILTERING & ARTIFACT
# --------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(sig, lowcut, highcut, fs):
    b,a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b,a,sig)

def apply_notch(sig, freq, fs, q):
    from scipy.signal import iirnotch
    nyq = fs / 2.0
    f0 = freq / nyq
    b,a = iirnotch(f0, q)
    return lfilter(b,a,sig)

def basic_artifact_removal(eeg_array, threshold=ARTIFACT_THRESHOLD):
    for ch in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch,:])>threshold):
            eeg_array[ch,:] = 0.0
    return eeg_array

# --------------------------
# BAND-POWER EXTRACTION
# --------------------------
def compute_band_power(sig, fs, band):
    fmin, fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs>=fmin)&(freqs<=fmax))[0]
    return np.mean(psd[idx])

def extract_bandpower_features(eeg_array):
    # shape (16, n_samples). We'll produce 64 features => 16 ch × 4 bands
    feats = []
    for ch in range(eeg_array.shape[0]):
        sig = eeg_array[ch,:]
        for (lo,hi) in BANDS.values():
            bp = compute_band_power(sig, SAMPLE_RATE, (lo,hi))
            feats.append(bp)
    return np.array(feats)

# --------------------------
# PARSE TRIALS
# We assume each .txt uses "sample number" in column 0,
# resets to 0 for each new trial, with ~10 trials per file.
# We'll produce ~10 trials => each 1 second => 256 samples
# Then we filter & compute band-power => 64-dim feature
# and label them with the file's label
# --------------------------
def load_data_from_file(filepath, label):
    print(f"[INFO] Loading data from {filepath} label={label}")
    df = pd.read_csv(filepath, sep=r'\s+|,|\t', engine='python', header=None)
    df = df.dropna(axis=1, how='all')
    if df.shape[1]<17:
        raise ValueError("Need at least 17 columns (sample# + 16 EEG). Found:", df.shape[1])

    arr = df.values
    trials = []
    current = []

    for row_idx in range(arr.shape[0]):
        row = arr[row_idx]
        sample_num = row[0]
        eeg_values = row[1:1+NUM_CHANNELS]  # columns 1..16

        if sample_num==0 and row_idx!=0:
            # means new trial started; finalize old trial
            if len(current)>0:
                trial_array = np.stack(current, axis=-1)  # (16, #samples)
                # trim/pad
                nsamp = trial_array.shape[1]
                if nsamp<DESIRED_SAMPLES:
                    pad = DESIRED_SAMPLES - nsamp
                    trial_array = np.pad(trial_array, ((0,0),(0,pad)), mode='constant')
                elif nsamp>DESIRED_SAMPLES:
                    trial_array = trial_array[:, -DESIRED_SAMPLES:]
                # filter per channel
                for ch in range(NUM_CHANNELS):
                    trial_array[ch,:] = apply_bandpass(trial_array[ch,:], BANDPASS_LOW,BANDPASS_HIGH,SAMPLE_RATE)
                    trial_array[ch,:] = apply_notch(trial_array[ch,:], NOTCH_FREQ,SAMPLE_RATE,NOTCH_Q)
                trial_array = basic_artifact_removal(trial_array)
                feats = extract_bandpower_features(trial_array)
                trials.append(feats)
            current=[]

        current.append(eeg_values)

    # handle last trial in file
    if len(current)>0:
        trial_array = np.stack(current, axis=-1)
        nsamp = trial_array.shape[1]
        if nsamp<DESIRED_SAMPLES:
            pad = DESIRED_SAMPLES - nsamp
            trial_array = np.pad(trial_array, ((0,0),(0,pad)), mode='constant')
        elif nsamp>DESIRED_SAMPLES:
            trial_array = trial_array[:, -DESIRED_SAMPLES:]
        for ch in range(NUM_CHANNELS):
            trial_array[ch,:] = apply_bandpass(trial_array[ch,:], BANDPASS_LOW,BANDPASS_HIGH,SAMPLE_RATE)
            trial_array[ch,:] = apply_notch(trial_array[ch,:], NOTCH_FREQ,SAMPLE_RATE,NOTCH_Q)
        trial_array = basic_artifact_removal(trial_array)
        feats = extract_bandpower_features(trial_array)
        trials.append(feats)

    X = np.array(trials)
    y = np.array([label]*len(trials))
    print(f"[INFO] Found {X.shape[0]} trials in {filepath}")
    return X,y


def main():
    all_X=[]
    all_y=[]

    for fname,label in FILE_LABELS:
        fpath = os.path.join(DATA_FOLDER,fname)
        X_, y_ = load_data_from_file(fpath, label)
        all_X.append(X_)
        all_y.append(y_)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print("[INFO] Final dataset shape:", X.shape, y.shape)

    # scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # simple train-test
    X_train, X_test, y_train, y_test = train_test_split(Xs,y, test_size=0.2, random_state=42, stratify=y)
    # train an SVM with probability
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    # retrain on full data if you want
    clf.fit(Xs, y)

    # Save
    model_path = "eeg_image_classifier.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(scaler, "scaler.joblib")
    print(f"[INFO] Model saved to {model_path}, scaler to scaler.joblib")


if __name__=="__main__":
    main()
