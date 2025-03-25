"""
realtime_classification_cnn.py

Revised streaming/inference code for a CNN-based EEG classifier.
Key modifications:
1. Loads a Keras CNN model saved as "eeg_cnn_model.h5".
2. Expects input trials to have DESIRED_SAMPLES (256) samples per channel and 16 channels,
   resulting in an input shape of (16, 256, 1).
3. Applies the same filtering pipeline (bandpass, notch, artifact removal) as during training.
4. Applies a saved StandardScaler (scaler.joblib) to the flattened trial vector and then reshapes it.
5. Supports Simulator Mode.

Install required packages:
    pip install numpy scipy joblib brainflow matplotlib tensorflow scikit-learn
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

import tensorflow as tf

# --- BrainFlow imports
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

from scipy.signal import butter, lfilter, iirnotch
from sklearn.preprocessing import StandardScaler

# ----------------------
# CONFIGURATION FLAGS & PARAMETERS
# ----------------------
SIMULATOR_MODE = True    # True to use simulated trial vectors instead of live data.
USE_IMAGES = False       # Set to True to load and display image files; otherwise display text.
MODEL_PATH = "eeg_cnn_model.h5"  # Keras CNN model file
SCALER_PATH = "scaler.joblib"    # StandardScaler saved during training
NUM_CLASSES = 12         # 12 classes
SAMPLE_RATE = 250        # Hz
DESIRED_SAMPLES = 256    # 256 samples per channel (expected input shape: 16 x 256)
REFRESH_INTERVAL = 1000  # milliseconds

# Artifact removal threshold
ARTIFACT_THRESHOLD = 1e7

# Filter parameters
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0

# ----------------------
# Simulated Trial Data
# ----------------------
# Representative base values (from your logs)
simulated_trial_base = np.array([
    -43632.35, -251079.49, -639686.37, -971181.05, -1019519.31,
    -851676.29, -681385.42, -615761.25, -616205.09, -599693.98,
    -533470.15, -452869.91, -402868.93, -378536.67, -342599.86, -286854.19,
    -238897.60, -210660.52, -180908.33, -136902.21
])
# Ensure simulated trial vector has exactly 16*DESIRED_SAMPLES features.
num_features = 16 * DESIRED_SAMPLES  # 4096 if DESIRED_SAMPLES = 256
num_repeats = int(np.ceil(num_features / simulated_trial_base.size))
simulated_trial_vector = np.tile(simulated_trial_base, num_repeats)[:num_features]

# ----------------------
# Filtering Helper Functions
# ----------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def apply_notch(data, notch_freq, fs, quality_factor):
    nyq = fs / 2.0
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    return lfilter(b, a, data)

def basic_artifact_removal(eeg_array, threshold=ARTIFACT_THRESHOLD):
    """
    Zero out an entire channel if any sample in that channel exceeds the threshold.
    """
    for ch_idx in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch_idx]) > threshold):
            print(f"[DEBUG] Channel {ch_idx} exceeds threshold {threshold}, zeroing out.")
            eeg_array[ch_idx, :] = 0.0
    return eeg_array

# ----------------------
# Real-Time Classification App using Tkinter after() scheduling
# ----------------------
class RealTimeClassifierApp:
    def __init__(self, master, model, scaler, class_labels, class_images, board, eeg_channels):
        """
        :param master: Tk root window.
        :param model: Loaded Keras CNN model.
        :param scaler: Loaded StandardScaler (or None).
        :param class_labels: List of class label strings.
        :param class_images: List of tk.PhotoImage objects (or None).
        :param board: BrainFlow board object (if not in simulator mode).
        :param eeg_channels: List of EEG channel indices.
        """
        self.master = master
        self.model = model
        self.scaler = scaler
        self.class_labels = class_labels
        self.class_images = class_images
        self.board = board
        self.eeg_channels = eeg_channels
        self.num_classes = len(class_labels)

        self.master.title("Real-Time EEG Classifier")

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_label = Label(self.master)
        self.image_label.pack(side=tk.BOTTOM)

        self.bars = self.ax.bar(range(self.num_classes), [0] * self.num_classes)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probability")
        self.ax.set_xlabel("Class")
        self.ax.set_title("Classification Probabilities")
        self.ax.set_xticks(range(self.num_classes))
        self.ax.set_xticklabels(self.class_labels, rotation=45, ha='right')

        self.master.after(REFRESH_INTERVAL, self.inference_iteration)

    def update_plot(self, probabilities):
        for bar, p in zip(self.bars, probabilities):
            bar.set_height(p)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        top_class = np.argmax(probabilities)
        if USE_IMAGES and self.class_images[top_class] is not None:
            self.image_label.config(image=self.class_images[top_class])
            self.image_label.image = self.class_images[top_class]
        else:
            self.image_label.config(text=f"Predicted: {self.class_labels[top_class]}", font=("Arial", 24))

    def inference_iteration(self):
        print("[INFO] Running inference iteration...")
        if SIMULATOR_MODE:
                # Instead of using the same base vector every time,
            # randomly perturb the base by sampling from a wider noise distribution.
            # For instance, add a random offset drawn from a uniform distribution,
            # and then add Gaussian noise.
            random_offset = np.random.uniform(-50000, 50000, size=simulated_trial_vector.shape)
            noise = np.random.normal(0, 50000.0, size=simulated_trial_vector.shape)
            trial_vector = simulated_trial_vector + random_offset + noise
            print(f"[INFO] trial_vector (first 20 values): {trial_vector[:20]}")
            print("[INFO] Using simulated trial vector.")
        else:
            count = self.board.get_board_data_count()
            print(f"[INFO] Board data count: {count}")
            if count <= 0:
                print("[WARNING] No data available from board yet.")
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return
            data = self.board.get_current_board_data(count)
            if data.size == 0:
                print("[WARNING] Received empty data array.")
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return
            print(f"[INFO] Data shape: {data.shape}")
            try:
                chunk = data[self.eeg_channels, :]
            except Exception as e:
                print("[ERROR] Unable to extract EEG channels:", e)
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return
            num_samples = chunk.shape[1]
            print(f"[INFO] Retrieved {num_samples} samples from board (EEG channels).")
            if num_samples < DESIRED_SAMPLES:
                print(f"[WARNING] Only {num_samples} samples available, need at least {DESIRED_SAMPLES}. Skipping iteration.")
                self.master.after(REFRESH_INTERVAL, self.inference_iteration)
                return
            if num_samples > DESIRED_SAMPLES:
                chunk = chunk[:, -DESIRED_SAMPLES:]
            elif num_samples < DESIRED_SAMPLES:
                pad_width = DESIRED_SAMPLES - num_samples
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
            channel_means = np.mean(chunk, axis=1)
            print(f"[INFO] Channel means: {channel_means}")
            for ch_idx in range(chunk.shape[0]):
                chunk[ch_idx, :] = apply_bandpass(chunk[ch_idx, :], BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
                chunk[ch_idx, :] = apply_notch(chunk[ch_idx, :], NOTCH_FREQ, SAMPLE_RATE, NOTCH_Q)
            chunk = basic_artifact_removal(chunk)
            # Flatten to 1D vector; note that training expects 16*DESIRED_SAMPLES features.
            trial_vector = chunk.reshape(1, -1)  
            print("[INFO] Trial Vector (first 20 values):", trial_vector[0][:20])
        
        # Apply normalization if scaler available.
        if self.scaler is not None:
            # Ensure trial_vector is 2D.
            if trial_vector.ndim == 1:
                trial_vector = trial_vector.reshape(1, -1)
            trial_vector = self.scaler.transform(trial_vector)
        # Reshape to CNN input shape: (1, 16, DESIRED_SAMPLES, 1)
        trial_vector = trial_vector.reshape(1, 16, DESIRED_SAMPLES, 1)
        
        # Use the CNN model for prediction.
        probs = self.model.predict(trial_vector)[0]
        print("[INFO] Predicted probabilities:", probs)
        self.update_plot(probs)
        self.master.after(REFRESH_INTERVAL, self.inference_iteration)

# ----------------------
# Main Function
# ----------------------
def main():
    print("[INFO] Loading trained model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    try:
        scaler = joblib.load(SCALER_PATH)
        print("[INFO] Scaler loaded from:", SCALER_PATH)
    except Exception as e:
        print("[WARNING] No scaler loaded. Proceeding without normalization.", e)
        scaler = None

    class_labels = [
        "Image1", "Image2", "Image3", "Image4", "Image5", "Image6",
        "Image7", "Image8", "Image9", "Image10", "Image11", "Image12"
    ]
    if USE_IMAGES:
        try:
            class_images = [
                tk.PhotoImage(file="image1.png"),
                tk.PhotoImage(file="image2.png"),
                tk.PhotoImage(file="image3.png"),
                tk.PhotoImage(file="image4.png"),
                tk.PhotoImage(file="image5.png"),
                tk.PhotoImage(file="image6.png"),
                tk.PhotoImage(file="image7.png"),
                tk.PhotoImage(file="image8.png"),
                tk.PhotoImage(file="image9.png"),
                tk.PhotoImage(file="image10.png"),
                tk.PhotoImage(file="image11.png"),
                tk.PhotoImage(file="image12.png")
            ]
        except tk.TclError as e:
            print("[ERROR] Unable to load one or more images. Check file paths.", e)
            class_images = [None] * NUM_CLASSES
    else:
        class_images = [None] * NUM_CLASSES

    if SIMULATOR_MODE:
        board = None
        eeg_channels = None
        print("[INFO] Simulator mode enabled. Skipping board initialization.")
    else:
        print("[INFO] Creating BrainFlow input params...")
        params = BrainFlowInputParams()
        params.ip_port = 6789
        params.ip_address = "192.168.4.1"
        board_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
        board = BoardShim(board_id, params)
        print("[INFO] Preparing session and configuring board...")
        board.prepare_session()
        try:
            board.config_board("~~")
        except Exception as e:
            print("[WARNING] Board config '~~' failed:", e)
        try:
            board.config_board("~6")
        except Exception as e:
            print("[WARNING] Board config '~6' failed:", e)
        try:
            board.config_board("//")
        except Exception as e:
            print("[WARNING] Board config '//' failed:", e)
        try:
            board.config_board("/4")
        except Exception as e:
            print("[WARNING] Board config '/4' failed:", e)
        for i in range(1, 9):
            try:
                board.config_board("x" + str(i) + "000000X")
            except Exception as e:
                print(f"[WARNING] Board config for channel {i} failed:", e)
        daisy_channels = "QWERTYUI"
        for i in range(0, 8):
            try:
                board.config_board("x" + daisy_channels[i] + "000000X")
            except Exception as e:
                print(f"[WARNING] Board config for daisy channel {daisy_channels[i]} failed:", e)
        board.start_stream(45000, "")
        print("[INFO] Board streaming started. Waiting 2 seconds to accumulate data...")
        time.sleep(2)
        board_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        print(f"[INFO] EEG Channels from BoardShim: {eeg_channels}")

    root = tk.Tk()
    app = RealTimeClassifierApp(root, model, scaler, class_labels, class_images, board, eeg_channels)
    print("[INFO] Entering Tkinter mainloop. Press Ctrl+C to exit.")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received. Exiting...")
    finally:
        if board is not None:
            print("[INFO] Stopping stream and releasing session.")
            board.stop_stream()
            board.release_session()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
