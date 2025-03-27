import time
import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import Label
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
# ----------------------
# CONFIG
# ----------------------
#conda config --env --set subdir osx-64 

SIMULATOR_MODE = True           # If True, we simulate sub-trials; If False, connect to board
USE_IMAGES = True

MODEL_PATH = "eeg_image_classifier.joblib"
SCALER_PATH = "scaler.joblib"

NUM_CLASSES = 12
SAMPLE_RATE = 250
SUBTRIAL_SAMPLES = 256
GROUP_SIZE = 4
NUM_CHANNELS = 16

REFRESH_INTERVAL = 200  # ms
ARTIFACT_THRESHOLD = 1e7

BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0

USE_BANDPOWER_FEATURES = True
USE_RAW_FLATTENING = False

BANDS = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta":  (13,30)
}

# ----------------------------------------------------------------
# Custom Classifiers (if your saved model uses them)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

class HybridRandomClassifier(BaseEstimator, ClassifierMixin):
    """
    A hybrid classifier that does real training (using a base estimator)
    but also injects a hashed-random component into the final probabilities.

    alpha=0 => all random, alpha=1 => pure trained model
    """
    def __init__(self, base_estimator=None, alpha=0.5, n_classes=12, stable=True):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.n_classes = n_classes
        self.stable = stable

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.base_estimator is None:
            from sklearn.linear_model import LogisticRegression
            self.base_estimator = LogisticRegression(max_iter=1000)
        self.base_estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        base_proba = self.base_estimator.predict_proba(X)  # (n_samples, n_classes)
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        for i in range(n_samples):
            row = X[i]
            row_bytes = row.tobytes()
            hash_val = hash(row_bytes)
            if self.stable:
                np.random.seed(hash_val % (2**32))
            else:
                ephemeral = np.random.randint(0, 2**31)
                combined_seed = (hash_val ^ ephemeral) % (2**32)
                np.random.seed(combined_seed)

            randvals = np.random.rand(self.n_classes)
            randvals /= randvals.sum()

            # Weighted blend
            blended = self.alpha * base_proba[i] + (1 - self.alpha) * randvals
            blended /= blended.sum()
            out[i] = blended

        return out

class HashedRandomClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that outputs unique random probabilities for each input row X,
    using the row's data as a seed.
    """
    def __init__(self, n_classes=12, stable=True):
        self.n_classes = n_classes
        self.stable = stable

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        for i in range(n_samples):
            row = X[i]
            row_bytes = row.tobytes()
            hash_val = hash(row_bytes)
            if self.stable:
                np.random.seed(hash_val % (2**32))
            else:
                ephemeral = np.random.randint(0, 2**31)
                combined_seed = (hash_val ^ ephemeral) % (2**32)
                np.random.seed(combined_seed)
            randvals = np.random.rand(self.n_classes)
            randvals /= randvals.sum()
            out[i] = randvals
        return out
# ----------------------------------------------------------------

try:
    import brainflow
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
except ImportError:
    print("[WARNING] brainflow not found. If using simulator mode, that's fine.")

from scipy.signal import butter, lfilter, iirnotch, welch
from sklearn.preprocessing import StandardScaler


# ----------------------
# NEW: CONTROL HOW RANDOM THE SIMULATED SUBTRIALS ARE IN SIM MODE
RANDOMNESS_PARAM = 0.6

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs/2.0
    low = lowcut / nyq
    high= highcut / nyq
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(sig, lowcut, highcut, fs):
    b,a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b,a,sig)

def apply_notch(sig, notch_freq, fs, q):
    nyq = fs / 2.0
    freq = notch_freq / nyq
    b,a = iirnotch(freq, q)
    return lfilter(b,a,sig)

def artifact_removal(eeg_array, threshold):
    for ch_idx in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch_idx]) > threshold):
            print(f"[DEBUG] artifact_removal => channel {ch_idx} zeroed out.")
            eeg_array[ch_idx,:] = 0.0
    return eeg_array

def compute_band_power(sig, fs, band):
    fmin,fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs>=fmin)&(freqs<=fmax))[0]
    if len(idx)==0:
        return 0.0
    return np.mean(psd[idx])

def extract_bandpower(eeg_chunk):
    feats=[]
    for ch in range(eeg_chunk.shape[0]):
        sig = eeg_chunk[ch,:]
        for (lo,hi) in BANDS.values():
            bp = compute_band_power(sig, SAMPLE_RATE,(lo,hi))
            feats.append(bp)
    return np.array(feats)

def extract_raw_flatten(eeg_chunk):
    return eeg_chunk.flatten()

def get_features(eeg_chunk):
    feat_list = []
    if USE_BANDPOWER_FEATURES:
        feat_list.append(extract_bandpower(eeg_chunk))
    if USE_RAW_FLATTENING:
        feat_list.append(extract_raw_flatten(eeg_chunk))

    if not feat_list:
        return np.zeros((1,))
    if len(feat_list)==1:
        return feat_list[0]
    else:
        return np.concatenate(feat_list)

# Global variables to hold the simulation data and current index
SIM_DATA = None
SIM_INDEX = 0

def load_sim_data():
    global SIM_DATA, SIM_INDEX
    try:
        SIM_DATA = np.loadtxt("SM001_0_3.txt")
        SIM_INDEX = 0
        print("[INFO] Loaded simulation data from sm001_1_3.txt with shape:", SIM_DATA.shape)
    except Exception as e:
        print(f"[ERROR] Could not load simulation data: {e}")
        SIM_DATA = None

def generate_subtrial_sim():
    """
    Instead of generating a random EEG subtrial, this version reads from the file
    'sm001_1_3.txt'. We assume:
      - Each row in the file is one sample.
      - Column 0 is time, and columns 1 to 16 (inclusive) are EEG channels.
    We read SUBTRIAL_SAMPLES rows, then transpose the data to match shape
    (NUM_CHANNELS, SUBTRIAL_SAMPLES).
    """
    global SIM_DATA, SIM_INDEX
    if SIM_DATA is None:
        load_sim_data()
        if SIM_DATA is None:
            # In case of failure, return a zero array
            return np.zeros((NUM_CHANNELS, SUBTRIAL_SAMPLES), dtype=float)
            
    total_samples = SIM_DATA.shape[0]
    # If there are not enough samples left, wrap around to the start
    if SIM_INDEX + SUBTRIAL_SAMPLES > total_samples:
        SIM_INDEX = 0
    # Extract rows for one subtrial and select EEG channels (columns 1 to 16)
    subtrial = SIM_DATA[SIM_INDEX:SIM_INDEX+SUBTRIAL_SAMPLES, 1:1+NUM_CHANNELS]
    SIM_INDEX += SUBTRIAL_SAMPLES
    # Transpose so that the shape becomes (NUM_CHANNELS, SUBTRIAL_SAMPLES)
    subtrial = subtrial.T
    return subtrial


# def generate_subtrial_sim():
#     """
#     produce a 16x256 sub-trial blending a fixed base wave with random draws,
#     scaled by RANDOMNESS_PARAM (0..1).
#     """
#     arr = np.zeros((NUM_CHANNELS, SUBTRIAL_SAMPLES), dtype=float)
#     t = np.arange(SUBTRIAL_SAMPLES)/SAMPLE_RATE

#     # Base wave for each channel
#     for ch in range(NUM_CHANNELS):
#         base_freq = 5.0 + ch
#         base_amp  = 1.0e4
#         base_phase= 0.0

#         # random freq, amp, phase
#         rand_freq  = np.random.uniform(2,45)
#         rand_amp   = np.random.uniform(8e3,1.2e4)
#         rand_phase = 2.0*np.pi*np.random.rand()

#         # blend freq, amp, phase
#         freq  = base_freq*(1-RANDOMNESS_PARAM) + rand_freq*(RANDOMNESS_PARAM)
#         amp   = base_amp*(1-RANDOMNESS_PARAM) + rand_amp*(RANDOMNESS_PARAM)
#         phase = base_phase*(1-RANDOMNESS_PARAM) + rand_phase*(RANDOMNESS_PARAM)

#         # add noise scaled by randomness
#         noise_scale = amp*0.2*RANDOMNESS_PARAM

#         wave = amp * np.sin(2*np.pi*freq*t + phase)
#         noise= np.random.normal(0, noise_scale, SUBTRIAL_SAMPLES)
#         arr[ch,:] = wave + noise

#     return arr

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

        self.subtrials = []

        self.master.title("Band-Power EEG Classifier (Group=4) - RealTime")

        # ----------------------------------------------------
        # 1) Figure for classification bar chart
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 2) (ADDED) Figure for raw EEG signals (only if not simulator)
        # ----------------------------------------------------
        self.eeg_fig = None
        self.eeg_ax = None
        self.eeg_canvas = None
        self.lines = []
        self.eeg_buffer = None

        if not SIMULATOR_MODE:
            self.eeg_fig, self.eeg_ax = plt.subplots(figsize=(8,5))
            self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=self.master)
            self.eeg_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Prepare a rolling buffer to store some recent samples
            # For example, store 2000 samples per channel for display
            self.buffer_size = 5000
            self.eeg_buffer = np.zeros((NUM_CHANNELS, self.buffer_size), dtype=float)

            # Plot 16 lines, each with an offset so they're stacked
            offsets = np.arange(NUM_CHANNELS) * 5e4  # vertical offset
            xvals = np.arange(self.buffer_size)

            for ch in range(NUM_CHANNELS):
                # We'll store the line object so we can update it
                line, = self.eeg_ax.plot(xvals, self.eeg_buffer[ch] + offsets[ch])
                self.lines.append(line)

            self.eeg_ax.set_xlim(0, self.buffer_size)
            # Y range from -some_offset to top
            self.eeg_ax.set_ylim(-1e5, offsets[-1] + 1e5)
            self.eeg_ax.set_xlabel("Samples")
            self.eeg_ax.set_ylabel("Amplitude (shifted per channel)")
            self.eeg_ax.set_title("EEG Channels (Most Recent Data)")

        # Start the periodic check
        self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)

    def update_plot(self, probs):
        """
        Update the classification bar chart & image
        """
        for bar, p in zip(self.bars, probs):
            bar.set_height(p)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        top_class = np.argmax(probs)
        label_text = f"Predicted: {self.class_labels[top_class]} (conf={probs[top_class]:.2f})"

        if USE_IMAGES and self.class_images[top_class] is not None:
            self.image_label.config(image=self.class_images[top_class])
            self.image_label.image = self.class_images[top_class]
        else:
            self.image_label.config(text=label_text, font=("Arial", 24))

    def update_eeg_plot(self, new_chunk):
        """
        (ADDED) Update the EEG rolling buffer and redraw the raw signals
        new_chunk shape = (NUM_CHANNELS, SUBTRIAL_SAMPLES)
        """
        # Shift the buffer to the left by SUBTRIAL_SAMPLES
        shift_amount = new_chunk.shape[1]
        self.eeg_buffer = np.roll(self.eeg_buffer, -shift_amount, axis=1)

        # Copy new data at the end
        self.eeg_buffer[:, -shift_amount:] = new_chunk

        # Redraw each channel with vertical offset
        offsets = np.arange(NUM_CHANNELS) * 5e4
        xvals = np.arange(self.buffer_size)
        for ch in range(NUM_CHANNELS):
            self.lines[ch].set_ydata(self.eeg_buffer[ch] + offsets[ch])

        self.eeg_fig.canvas.draw()
        self.eeg_fig.canvas.flush_events()

    def process_big_trial(self, big_arr):
        print("[DEBUG] process_big_trial => shape=", big_arr.shape)
        # filter
        # for ch in range(NUM_CHANNELS):
        #     big_arr[ch,:] = apply_bandpass(big_arr[ch,:], BANDPASS_LOW,BANDPASS_HIGH,SAMPLE_RATE)
        #     big_arr[ch,:] = apply_notch(big_arr[ch,:], NOTCH_FREQ,SAMPLE_RATE,NOTCH_Q)
        # big_arr = artifact_removal(big_arr, ARTIFACT_THRESHOLD)

        feats = get_features(big_arr).reshape(1,-1)
        if self.scaler:
            feats = self.scaler.transform(feats)

        # Some classifiers do not have predict_proba, fallback to predict
        try:
            probs = self.model.predict_proba(feats)[0]
        except AttributeError:
            pred_lbl = self.model.predict(feats)[0]
            probs = np.zeros(self.num_classes)
            probs[pred_lbl] = 1.0

        print("[DEBUG] Classification features =>", feats)
        print("[DEBUG] Classification probabilities =>", probs)
        return probs

    def check_for_subtrial(self):
        if SIMULATOR_MODE:
            sub = generate_subtrial_sim()
            self.subtrials.append(sub)
            print(f"[DEBUG] simulator => new sub-trial. total={len(self.subtrials)}")
        else:
            # real board
            if self.board is None or self.eeg_channels is None:
                print("[ERROR] BrainFlow board not configured.")
                self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)
                return

            c = self.board.get_board_data_count()
            print(f"[DEBUG] board data count={c}")
            if c < SUBTRIAL_SAMPLES:
                print("[DEBUG] Not enough data => skip.")
                self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)
                return
            data = self.board.get_current_board_data(c)
            chunk = data[self.eeg_channels, -SUBTRIAL_SAMPLES:]
            self.subtrials.append(chunk)
            print(f"[DEBUG] real board => new sub-trial. total={len(self.subtrials)}")

            # (ADDED) Update the live EEG plot with just-arrived chunk
            if self.eeg_ax is not None:
                self.update_eeg_plot(chunk)

        if len(self.subtrials) >= GROUP_SIZE:
            big_arr = np.concatenate(self.subtrials, axis=1)
            self.subtrials = []
            print(f"[INFO] formed a big trial => shape={big_arr.shape}, now classify.")
            probs = self.process_big_trial(big_arr)
            self.update_plot(probs)
        else:
            needed = GROUP_SIZE - len(self.subtrials)
            print(f"[DEBUG] waiting for {needed} more sub-trials...")

        self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)

def load_class_images(num_classes, master=None):
    images = [None]*num_classes
    for i in range(num_classes):
        fn = f"image{i+1}.png"
        if os.path.exists(fn):
            pil_img = Image.open(fn)
            # "Zoom out" => resize to 400 wide x 200 high
            pil_img = pil_img.resize((400, 200))
            images[i] = ImageTk.PhotoImage(pil_img, master=master)
        else:
            print(f"[WARNING] No image file => {fn}, skipping.")
    return images

def main():
    print("[INFO] Loading model from:", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found => {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)

    scaler = None
    if os.path.exists(SCALER_PATH):
        print("[INFO] Loading scaler from:", SCALER_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"[WARNING] No scaler found => {SCALER_PATH}, continuing without scaling.")

    class_labels = [f"Class {i}" for i in range(NUM_CLASSES)]

    root = tk.Tk()
    root.title("EEG Classification GUI")

    # Load images after root is created
    class_images = load_class_images(NUM_CLASSES, master=root)

    board=None
    eeg_channels=None
    if not SIMULATOR_MODE:
        print("[INFO] Setting up BrainFlow board (Cyton Daisy example).")
        params = brainflow.board_shim.BrainFlowInputParams()
        params.ip_port=6789
        params.ip_address="192.168.4.1"
        board_id=BoardIds.CYTON_DAISY_WIFI_BOARD.value
        board=BoardShim(board_id, params)
        board.prepare_session()
        board.config_board("~~")
        board.config_board("~6")

        # Check board mode, change to Marker mode
        board.config_board("//")
        board.config_board("/4")


        for i in range(1,9):
            board.config_board("x" + str(i) + "000000X")

        daisy_channels = "QWERTYUI"

        for i in range(0,8):
            board.config_board("x" + daisy_channels[i] + "000000X")

        board.start_stream()
        time.sleep(2)
        eeg_channels=BoardShim.get_eeg_channels(board_id)
        print("[INFO] EEG channels =>", eeg_channels)

    app = RealTimeClassifierApp(
        master=root,
        model=model,
        scaler=scaler,
        class_labels=class_labels,
        class_images=class_images,
        board=board,
        eeg_channels=eeg_channels
    )

    print(f"[INFO] Entering mainloop.. (RANDOMNESS_PARAM={RANDOMNESS_PARAM})")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[INFO] user interrupted.")
    finally:
        if board is not None:
            board.stop_stream()
            board.release_session()

if __name__=="__main__":
    main()
