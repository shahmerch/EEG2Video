"""
bandpower_classification.py

Real-time (or simulated) classification script for a scikit-learn model
(SVM or logistic regression) trained on band-power features (delta/theta/alpha/beta).
Addresses the "constant probabilities" bug in simulator mode by generating
more diverse synthetic data each iteration.

1. Loads "eeg_bandpower_model.joblib" (SVM or logistic regression) plus "scaler.joblib".
2. If SIMULATOR_MODE is True, it generates a random chunk (16 x 256) each iteration
   that includes a random sinusoid in the 1–50 Hz range plus some noise, ensuring
   more variety in band-power distributions.
3. If not simulator mode, reads from BoardShim, filters and computes band-power.
4. Displays probabilities in a Tkinter GUI.

Install needed packages:
  pip install numpy scipy joblib brainflow matplotlib scikit-learn
"""

import time
import numpy as np
import joblib
import tkinter as tk
from tkinter import Label
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from scipy.signal import butter, lfilter, iirnotch, welch
from sklearn.preprocessing import StandardScaler

# ----------------------
# CONFIGURATION
# ----------------------
SIMULATOR_MODE = True
USE_IMAGES = False
MODEL_PATH = "eeg_image_classifier.joblib"
SCALER_PATH = "scaler.joblib"

NUM_CLASSES = 12
SAMPLE_RATE = 250
DESIRED_SAMPLES = 256
NUM_CHANNELS = 16
REFRESH_INTERVAL = 1000  # ms

ARTIFACT_THRESHOLD = 1e7

BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0

# Frequency bands from training
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30)
}

# ----------------------
# FIXED: More Varied Simulation
# ----------------------
def generate_random_chunk():
    """
    Produce a random 16x256 chunk, each channel = sinusoid at a random freq + noise,
    ensuring more varied band-power each iteration.
    """
    chunk = np.zeros((NUM_CHANNELS, DESIRED_SAMPLES), dtype=float)
    t = np.arange(DESIRED_SAMPLES) / SAMPLE_RATE
    for ch in range(NUM_CHANNELS):
        freq = np.random.uniform(2.0, 120.0)  # random freq in 2–45 Hz
        phase = 2.0 * np.pi * np.random.rand()
        amplitude = np.random.uniform(0.5e4, 2e4)  # random amplitude in microvolts
        # sinusoid + small noise
        chunk[ch,:] = amplitude * np.sin(2*np.pi*freq*t + phase) \
                      + np.random.normal(0, amplitude*0.3, DESIRED_SAMPLES)
        
    return chunk


# ----------------------
# FILTERING & ARTIFACT
# ----------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(sig, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, sig)

def apply_notch(sig, notch_freq, fs, q):
    from scipy.signal import iirnotch
    nyq = fs / 2.0
    freq = notch_freq / nyq
    b, a = iirnotch(freq, q)
    return lfilter(b, a, sig)

def artifact_removal(eeg_array, threshold):
    for ch_idx in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch_idx]) > threshold):
            eeg_array[ch_idx, :] = 0.0
    return eeg_array


# ----------------------
# BAND-POWER FEATURES
# ----------------------
def compute_band_power(sig, fs, band):
    fmin, fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    return np.mean(psd[idx])

def extract_features(eeg_chunk):
    """
    band-power across delta/theta/alpha/beta, shape (16,256)->(64,)
    """
    feats = []
    for ch in range(eeg_chunk.shape[0]):
        sig = eeg_chunk[ch, :]
        for band_name, (lowf, highf) in BANDS.items():
            bp = compute_band_power(sig, SAMPLE_RATE, (lowf, highf))
            feats.append(bp)
    return np.array(feats)  # (64,)


# ----------------------
# TKINTER CLASSIFIER
# ----------------------
class RealTimeClassifierApp:
    def __init__(self, master, model, scaler, class_labels, class_images, board, eeg_channels):
        self.master = master
        self.model = model
        self.scaler = scaler
        self.class_labels = class_labels
        self.class_images = class_images
        self.board = board
        self.eeg_channels = eeg_channels
        self.num_classes = len(class_labels)

        self.master.title("Band-Power EEG Classifier")

        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.master)
        self.image_label.pack(side=tk.BOTTOM)

        self.bars = self.ax.bar(range(self.num_classes), [0]*self.num_classes)
        self.ax.set_ylim(0,1)
        self.ax.set_ylabel("Probability")
        self.ax.set_xlabel("Class")
        self.ax.set_title("Classification Probabilities")
        self.ax.set_xticks(range(self.num_classes))
        self.ax.set_xticklabels(self.class_labels, rotation=45, ha='right')

        self.master.after(REFRESH_INTERVAL, self.inference_iteration)

    def update_plot(self, probs):
        for bar, p in zip(self.bars, probs):
            bar.set_height(p)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        top_class = np.argmax(probs)
        if USE_IMAGES and self.class_images[top_class] is not None:
            self.image_label.config(image=self.class_images[top_class])
            self.image_label.image = self.class_images[top_class]
        else:
            self.image_label.config(text=f"Predicted: {self.class_labels[top_class]}", font=("Arial", 24))

    def inference_iteration(self):
        print("[INFO] Inference iteration...")
        if SIMULATOR_MODE:
            chunk = generate_random_chunk()
            print("[INFO] Using random simulated chunk. shape:", chunk.shape)
        else:
            count = self.board.get_board_data_count()
            print("[INFO] Board data count:", count)
            if count < DESIRED_SAMPLES:
                print("[WARNING] Not enough data, skipping.")
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return
            data = self.board.get_board_data()  # or get_current_board_data(count)
            if data.size == 0:
                print("[WARNING] Empty data array, skipping.")
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return
            print("[INFO] data shape:", data.shape)
            try:
                chunk = data[self.eeg_channels, -DESIRED_SAMPLES:]
            except Exception as e:
                print("[ERROR] slice EEG channels:", e)
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return

        # Filter & artifact
        for ch in range(chunk.shape[0]):
            chunk[ch,:] = apply_bandpass(chunk[ch,:], BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
            chunk[ch,:] = apply_notch(chunk[ch,:], NOTCH_FREQ, SAMPLE_RATE, NOTCH_Q)
        chunk = artifact_removal(chunk, ARTIFACT_THRESHOLD)

        # Extract band-power features
        feats = extract_features(chunk)
        feats = feats.reshape(1, -1)

        # scale
        if self.scaler is not None:
            feats = self.scaler.transform(feats)

        # predict
        try:
            probs = self.model.predict_proba(feats)[0]
        except AttributeError:
            pred_label = self.model.predict(feats)[0]
            probs = np.zeros(self.num_classes)
            probs[pred_label] = 1.0
        print("[INFO] Features:", feats)
        print("[INFO] Probabilities:", probs)
        self.update_plot(probs)
        self.master.after(REFRESH_INTERVAL, self.inference_iteration)


def main():
    print("[INFO] Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    try:
        scaler = joblib.load(SCALER_PATH)
        print("[INFO] Loaded scaler from:", SCALER_PATH)
    except Exception as e:
        print("[WARNING] No scaler, continuing without. Error:", e)
        scaler = None

    class_labels = [f"Image{i}" for i in range(1, NUM_CLASSES+1)]
    if USE_IMAGES:
        class_images = [None]*NUM_CLASSES
        # TODO: load your images
    else:
        class_images = [None]*NUM_CLASSES

    if SIMULATOR_MODE:
        board = None
        eeg_channels = None
        print("[INFO] Simulator mode ON, skipping board init.")
    else:
        print("[INFO] Setting up BrainFlow board.")
        params = BrainFlowInputParams()
        params.ip_port = 6789
        params.ip_address = "192.168.4.1"
        board_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(2)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        print("[INFO] EEG channels:", eeg_channels)

    root = tk.Tk()
    app = RealTimeClassifierApp(root, model, scaler, class_labels, class_images, board, eeg_channels)
    print("[INFO] Entering Tkinter mainloop.")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt, shutting down.")
    finally:
        if board is not None:
            board.stop_stream()
            board.release_session()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
