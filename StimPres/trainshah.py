"""
train_model.py

This script loads 12 files (each containing 10 trials) of OpenBCI EEG data,
applies bandpass and notch filtering plus basic artifact removal, then
ensures each trial has a fixed length (DESIRED_TRIAL_LENGTH samples per trial)
by trimming or padding, and finally flattens each trial into a feature vector.
It then normalizes the features using a StandardScaler, trains a RandomForestClassifier,
evaluates performance, and saves both the model and scaler.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------
# FILTERING CONFIGURATIONS
# ------------------------
SAMPLE_RATE = 250
NYQUIST = SAMPLE_RATE / 2
BANDPASS_LOW = 1.0    # 1 Hz
BANDPASS_HIGH = 50.0  # 50 Hz
NOTCH_FREQ = 60.0     # 60 Hz
Q_FACTOR = 30.0       # Quality factor for notch filter

# Set the desired trial length (number of samples per trial).
DESIRED_TRIAL_LENGTH = 256  # We'll use 256 samples (so feature vector = 16*256 = 4096)

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Creates a Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs):
    """Applies bandpass filter to 1D numpy array 'data'."""
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def apply_notch(data, notch_freq, fs, quality_factor):
    """Applies a notch filter (IIR) to remove line noise at 'notch_freq' Hz."""
    nyq = fs / 2.0
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    return lfilter(b, a, data)

def basic_artifact_removal(eeg_array, threshold=150e-6):
    """
    Simple artifact removal placeholder.
    If any sample in a channel exceeds the threshold (in absolute value),
    that entire channel is set to 0. (Adjust threshold as needed.)
    """
    for ch_idx in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch_idx]) > threshold):
            eeg_array[ch_idx, :] = 0.0
    return eeg_array

def load_data_from_file(filepath, label):
    """
    Loads one file's data (10 trials) and returns feature matrix X and label vector y.
    Each trial is processed to have DESIRED_TRIAL_LENGTH samples (for 16 channels).
    """
    print(f"[INFO] Loading data from file: {filepath} with label: {label}")
    # Adjust delimiter as needed (here we try multiple common delimiters)
    df = pd.read_csv(filepath, sep='\t|,| +', engine='python', header=None)
    df = df.dropna(axis=1, how='all')  # Drop empty columns if needed

    if df.shape[1] < 17:
        raise ValueError("File does not have enough columns for sample + 16 EEG channels.")

    data_np = df.values
    trials = []
    current_trial = []

    for row_idx in range(data_np.shape[0]):
        row = data_np[row_idx]
        sample_num = row[0]
        eeg_values = row[1:17]  # columns 1..16

        # When sample number resets (and not first row), finish the previous trial.
        if sample_num == 0 and row_idx != 0:
            if len(current_trial) > 0:
                trial_array = np.stack(current_trial, axis=-1)  # shape: (16, n_samples)
                # Process the trial to ensure a fixed length:
                if trial_array.shape[1] < DESIRED_TRIAL_LENGTH:
                    # Pad with zeros
                    pad_width = DESIRED_TRIAL_LENGTH - trial_array.shape[1]
                    trial_array = np.pad(trial_array, ((0,0), (0, pad_width)), mode='constant')
                elif trial_array.shape[1] > DESIRED_TRIAL_LENGTH:
                    # Trim to last DESIRED_TRIAL_LENGTH samples
                    trial_array = trial_array[:, -DESIRED_TRIAL_LENGTH:]
                # Append the processed trial.
                trials.append(trial_array)
            current_trial = []
        current_trial.append(eeg_values)

    # Process the last trial in the file.
    if len(current_trial) > 0:
        trial_array = np.stack(current_trial, axis=-1)
        if trial_array.shape[1] < DESIRED_TRIAL_LENGTH:
            pad_width = DESIRED_TRIAL_LENGTH - trial_array.shape[1]
            trial_array = np.pad(trial_array, ((0,0), (0, pad_width)), mode='constant')
        elif trial_array.shape[1] > DESIRED_TRIAL_LENGTH:
            trial_array = trial_array[:, -DESIRED_TRIAL_LENGTH:]
        trials.append(trial_array)

    print(f"[INFO] Found {len(trials)} trials in {filepath}")

    # Apply filtering and artifact removal for each trial.
    filtered_trials = []
    for trial_data in trials:
        # trial_data is shape (16, DESIRED_TRIAL_LENGTH)
        for ch in range(trial_data.shape[0]):
            trial_data[ch, :] = apply_bandpass(trial_data[ch, :], BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
            trial_data[ch, :] = apply_notch(trial_data[ch, :], NOTCH_FREQ, SAMPLE_RATE, Q_FACTOR)
        trial_data = basic_artifact_removal(trial_data)
        filtered_trials.append(trial_data)

    # Flatten each trial to create a feature vector.
    X = np.array([t.reshape(-1) for t in filtered_trials])  # Each vector now has 16*DESIRED_TRIAL_LENGTH features.
    y = np.array([label] * len(filtered_trials))
    return X, y

def main():
    data_folder = "./eeg_data"
    file_labels = [
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
        ("SM001_11_3.txt", 11)
    ]

    all_X = []
    all_y = []
    for fname, lbl in file_labels:
        filepath = os.path.join(data_folder, fname)
        X, y = load_data_from_file(filepath, lbl)
        print(f"[INFO] Loaded {X.shape[0]} trials from {fname}")
        all_X.append(X)
        all_y.append(y)

    if all_X:
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
    else:
        raise ValueError("No trials loaded from any file.")

    print(f"[INFO] Dataset shape: X={X.shape}, y={y.shape}")

    # Split the dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("[INFO] Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[INFO] Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print("[INFO] Training complete.")
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # Save the model and the scaler.
    joblib.dump(clf, "eeg_image_classifier.joblib")
    print("[INFO] Model saved to eeg_image_classifier.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("[INFO] Scaler saved to scaler.joblib")

if __name__ == "__main__":
    main()
