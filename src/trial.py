import time
import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import Label
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# ----------------------
# CONFIG
SIMULATOR_MODE = True           # If True, simulate sub-trials; if False, connect to board
SIM_FILE = "./All Nos/Nos000/Nos000_7_3.txt"
USE_IMAGES = True

MODELS_DIR = "models"           # Directory where separate binary classifiers are saved
SCALER_PATH = "scaler.joblib"

NUM_CLASSES = 12
SAMPLE_RATE = 250
SUBTRIAL_SAMPLES = 250          # Should match training
GROUP_SIZE = 4
NUM_CHANNELS = 16

REFRESH_INTERVAL = 200  # ms

# Use same filter parameters as training
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0

# Use the same artifact threshold as training
ARTIFACT_THRESHOLD = 150e-6

USE_BANDPOWER_FEATURES = True
USE_RAW_FLATTENING = False

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30)
}

# ------------------ CUSTOM CLASSIFIERS (for loading models) ------------------
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

class HybridRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, alpha=0.5, n_classes=12, stable=False):
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
        base_proba = self.base_estimator.predict_proba(X)
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        for i in range(n_samples):
            row = X[i]
            row_bytes = row.tobytes()
            hash_val = hash(row_bytes)
            if self.stable:
                seed = hash_val % (2**32)
            else:
                ephemeral = np.random.randint(0, 2**31)
                seed = (hash_val ^ ephemeral) % (2**32)
            rng = np.random.RandomState(seed)
            randvals = rng.rand(self.n_classes)
            randvals /= randvals.sum()
            blended = self.alpha * base_proba[i] + (1 - self.alpha) * randvals
            blended /= blended.sum()
            out[i] = blended
        return out

class HashedRandomClassifier(BaseEstimator, ClassifierMixin):
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
                seed = hash_val % (2**32)
            else:
                ephemeral = np.random.randint(0, 2**31)
                seed = (hash_val ^ ephemeral) % (2**32)
            rng = np.random.RandomState(seed)
            randvals = rng.rand(self.n_classes)
            randvals /= randvals.sum()
            out[i] = randvals
        return out

# ------------------ SIGNAL PROCESSING & FEATURE EXTRACTION ------------------
from scipy.signal import butter, lfilter, iirnotch, welch
from sklearn.preprocessing import StandardScaler

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(sig, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, sig)

def apply_notch(sig, notch_freq, fs, q):
    nyq = fs / 2.0
    freq = notch_freq / nyq
    b, a = iirnotch(freq, q)
    return lfilter(b, a, sig)

def artifact_removal(eeg_array, threshold):
    for ch in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch, :]) > threshold):
            print(f"[DEBUG] artifact_removal => channel {ch} zeroed out.")
            eeg_array[ch, :] = 0.0
    return eeg_array

def compute_band_power(sig, fs, band):
    fmin, fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    return np.mean(psd[idx]) if len(idx) > 0 else 0.0

def extract_bandpower(eeg_chunk):
    feats = []
    for ch in range(eeg_chunk.shape[0]):
        sig = eeg_chunk[ch, :]
        for (lo, hi) in BANDS.values():
            feats.append(compute_band_power(sig, SAMPLE_RATE, (lo, hi)))
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
    if len(feat_list) == 1:
        return feat_list[0]
    else:
        return np.concatenate(feat_list)

# ------------------ SIMULATION DATA ------------------
SIM_DATA = None
SIM_INDEX = 0

def load_sim_data():
    global SIM_DATA, SIM_INDEX
    if os.path.exists(SIM_FILE):
        try:
            SIM_DATA = np.loadtxt(SIM_FILE)
            SIM_INDEX = 0
            print("[INFO] Loaded simulation data with shape:", SIM_DATA.shape)
        except Exception as e:
            print(f"[ERROR] Could not load simulation data: {e}")
            SIM_DATA = None
    else:
        print(f"[ERROR] Simulation file {SIM_FILE} not found. Generating random simulation data.")
        num_samples = 1024
        sample_nums = np.arange(num_samples).reshape(-1, 1)
        eeg_vals = np.random.uniform(-1e-4, 1e-4, size=(num_samples, NUM_CHANNELS))
        SIM_DATA = np.concatenate([sample_nums, eeg_vals], axis=1)
        SIM_INDEX = 0
        print("[INFO] Generated random simulation data with shape:", SIM_DATA.shape)

def generate_subtrial_sim():
    """
    Extracts a subtrial from the simulation data in the same manner as training.
    Assumes column 0 is the sample number and columns 1 to NUM_CHANNELS are EEG channels.
    """
    global SIM_DATA, SIM_INDEX
    if SIM_DATA is None:
        load_sim_data()
        if SIM_DATA is None:
            return np.random.uniform(-1e-4, 1e-4, size=(NUM_CHANNELS, SUBTRIAL_SAMPLES))
    total_samples = SIM_DATA.shape[0]
    if SIM_INDEX + SUBTRIAL_SAMPLES > total_samples:
        SIM_INDEX = 0
    subtrial = SIM_DATA[SIM_INDEX:SIM_INDEX+SUBTRIAL_SAMPLES, 1:1+NUM_CHANNELS]
    SIM_INDEX += SUBTRIAL_SAMPLES
    return subtrial.T

# ------------------ REALTIME GUI APPLICATION ------------------
class RealTimeClassifierApp:
    def __init__(self, master, models, scaler, class_labels, class_img_paths, board, eeg_channels):
        self.master = master
        self.models = models  # dictionary mapping class index to binary classifier
        self.scaler = scaler
        self.class_labels = class_labels
        self.board = board
        self.eeg_channels = eeg_channels
        self.num_classes = len(class_labels)
        self.subtrials = []
        self.master.title("Band-Power EEG Classifier (Group=4) - RealTime")
        # Load class images both as PhotoImages (for the predicted image) and as NumPy arrays for the image bars:
        self.class_photo_images = []
        self.class_img_arrays = []
        for path in class_img_paths:
            if os.path.exists(path):
                pil_img = Image.open(path).resize((80, 80))  # image for bar chart
                self.class_photo_images.append(ImageTk.PhotoImage(pil_img, master=self.master))
                # Also load a larger version for the predicted display
                large_img = Image.open(path).resize((200, 200))
                self.class_img_arrays.append(np.array(large_img))
            else:
                self.class_photo_images.append(None)
                self.class_img_arrays.append(None)
        # --- Create an axes for the image bar chart ---
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_xlim(0, self.num_classes)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks(np.arange(self.num_classes) + 0.4)
        self.ax.set_xticklabels(self.class_labels, rotation=45, ha='right')
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Classification Probabilities (Image Bars)")
        # Create one image artist per class for the bar chart.
        self.img_artists = []
        for i in range(self.num_classes):
            if self.class_img_arrays[i] is not None:
                artist = self.ax.imshow(self.class_img_arrays[i],
                                        extent=[i, i+0.8, 0, 0],
                                        aspect='auto')
            else:
                artist = None
            self.img_artists.append(artist)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # --- Predicted Class Display ---
        self.predicted_frame = tk.Frame(self.master)
        self.predicted_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.predicted_image_label = tk.Label(self.predicted_frame)
        self.predicted_image_label.pack(side=tk.LEFT, padx=10)
        self.predicted_text_label = tk.Label(self.predicted_frame, font=("Arial", 24))
        self.predicted_text_label.pack(side=tk.LEFT, padx=10)
        # --- (Optional) Raw EEG Plot ---
        self.eeg_fig = None
        self.eeg_ax = None
        self.eeg_canvas = None
        self.lines = []
        self.eeg_buffer = None
        if not SIMULATOR_MODE:
            self.eeg_fig, self.eeg_ax = plt.subplots(figsize=(8, 5))
            self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=self.master)
            self.eeg_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.buffer_size = 5000
            self.eeg_buffer = np.zeros((NUM_CHANNELS, self.buffer_size), dtype=float)
            offsets = np.arange(NUM_CHANNELS) * 5e4
            xvals = np.arange(self.buffer_size)
            for ch in range(NUM_CHANNELS):
                line, = self.eeg_ax.plot(xvals, self.eeg_buffer[ch] + offsets[ch])
                self.lines.append(line)
            self.eeg_ax.set_xlim(0, self.buffer_size)
            self.eeg_ax.set_ylim(-1e5, offsets[-1] + 1e5)
            self.eeg_ax.set_xlabel("Samples")
            self.eeg_ax.set_ylabel("Amplitude (shifted per channel)")
            self.eeg_ax.set_title("EEG Channels (Most Recent Data)")
        self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)

    def update_image_bars(self, probs):
        # Update the image bar extents
        for i in range(self.num_classes):
            artist = self.img_artists[i]
            if artist is not None:
                artist.set_extent([i, i+0.8, 0, probs[i]])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Determine predicted class
        top_class = int(np.argmax(probs))
        conf = probs[top_class]
        # Update the predicted image and text
        if self.class_photo_images[top_class] is not None:
            self.predicted_image_label.config(image=self.class_photo_images[top_class])
            self.predicted_image_label.image = self.class_photo_images[top_class]
        self.predicted_text_label.config(text=f"Predicted: {self.class_labels[top_class]} (conf={conf:.2f})")

    def update_eeg_plot(self, new_chunk):
        shift_amount = new_chunk.shape[1]
        self.eeg_buffer = np.roll(self.eeg_buffer, -shift_amount, axis=1)
        self.eeg_buffer[:, -shift_amount:] = new_chunk
        offsets = np.arange(NUM_CHANNELS) * 5e4
        xvals = np.arange(self.buffer_size)
        for ch in range(NUM_CHANNELS):
            self.lines[ch].set_ydata(self.eeg_buffer[ch] + offsets[ch])
        self.eeg_fig.canvas.draw()
        self.eeg_fig.canvas.flush_events()

    def process_big_trial(self, big_arr):
        print("[DEBUG] process_big_trial => shape=", big_arr.shape)
        feats = get_features(big_arr).reshape(1, -1)
        if self.scaler:
            feats = self.scaler.transform(feats)
        probs = np.zeros(self.num_classes)
        print(f"[DEBUG] Classifier input: {feats}")
        for cls, model in self.models.items():
            try:
                prob_arr = model.predict_proba(feats)[0]
                if 1 in model.classes_:
                    idx = list(model.classes_).index(1)
                    p = prob_arr[idx]
                else:
                    p = 0.0
                probs[cls] = p
                print(f"[DEBUG] Classifier {cls} positive probability: {p:.4f}")
            except AttributeError:
                pred = model.predict(feats)[0]
                probs[cls] = 1.0 if pred == 1 else 0.0
        print("[DEBUG] Aggregated classification probabilities:", probs)
        return probs

    def check_for_subtrial(self):
        if SIMULATOR_MODE:
            sub = generate_subtrial_sim()
            self.subtrials.append(sub)
            print(f"[DEBUG] Simulator => new sub-trial. Total collected: {len(self.subtrials)}")
        else:
            if self.board is None or self.eeg_channels is None:
                print("[ERROR] BrainFlow board not configured.")
                try:
                    self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)
                except tk.TclError:
                    pass
                return
            c = self.board.get_board_data_count()
            print(f"[DEBUG] Board data count={c}")
            if c < SUBTRIAL_SAMPLES:
                print("[DEBUG] Not enough data => skip.")
                try:
                    self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)
                except tk.TclError:
                    pass
                return
            data = self.board.get_current_board_data(c)
            chunk = data[self.eeg_channels, -SUBTRIAL_SAMPLES:]
            self.subtrials.append(chunk)
            print(f"[DEBUG] Real board => new sub-trial. Total collected: {len(self.subtrials)}")
            if self.eeg_ax is not None:
                self.update_eeg_plot(chunk)
        if len(self.subtrials) >= GROUP_SIZE:
            big_arr = np.concatenate(self.subtrials, axis=1)
            self.subtrials = []
            print(f"[INFO] Formed a big trial => shape={big_arr.shape}. Now classifying.")
            probs = self.process_big_trial(big_arr)
            self.update_image_bars(probs)
        else:
            needed = GROUP_SIZE - len(self.subtrials)
            print(f"[DEBUG] Waiting for {needed} more sub-trials...")
        try:
            self.master.after(REFRESH_INTERVAL, self.check_for_subtrial)
        except tk.TclError:
            pass

def load_class_images(num_classes, master=None):
    # Return list of image file paths.
    paths = []
    for i in range(num_classes):
        path = f"./images/image{i+1}.png"
        paths.append(path)
    return paths

def main():
    print("[INFO] Loading binary classifier models from:", MODELS_DIR)
    models = {}
    if not os.path.exists(MODELS_DIR):
        print(f"[ERROR] Models directory not found => {MODELS_DIR}")
        return
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("eeg_image_classifier_class_") and fname.endswith(".joblib"):
            cls_num = int(fname[len("eeg_image_classifier_class_"):-len(".joblib")])
            model_path = os.path.join(MODELS_DIR, fname)
            models[cls_num] = joblib.load(model_path)
            print(f"[INFO] Loaded classifier for class {cls_num} from {model_path}")
    if len(models) == 0:
        print("[ERROR] No classifiers loaded. Exiting.")
        return

    scaler = None
    if os.path.exists(SCALER_PATH):
        print("[INFO] Loading scaler from:", SCALER_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"[WARNING] No scaler found at {SCALER_PATH}, continuing without scaling.")

    class_labels = [f"Class {i}" for i in range(NUM_CLASSES)]
    class_img_paths = load_class_images(NUM_CLASSES)
    root = tk.Tk()
    root.title("EEG Classification GUI")
    board = None
    eeg_channels = None
    if not SIMULATOR_MODE:
        try:
            import brainflow
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        except ImportError:
            print("[ERROR] BrainFlow not found.")
            return
        print("[INFO] Setting up BrainFlow board.")
        params = BrainFlowInputParams()
        params.ip_port = 6789
        params.ip_address = "192.168.4.1"
        board_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.config_board("~~")
        board.config_board("~6")
        board.config_board("//")
        board.config_board("/4")
        for i in range(1, 9):
            board.config_board("x" + str(i) + "000000X")
        daisy_channels = "QWERTYUI"
        for i in range(0, 8):
            board.config_board("x" + daisy_channels[i] + "000000X")
        board.start_stream()
        time.sleep(2)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        print("[INFO] EEG channels =>", eeg_channels)

    app = RealTimeClassifierApp(
        master=root,
        models=models,
        scaler=scaler,
        class_labels=class_labels,
        class_img_paths=class_img_paths,
        board=board,
        eeg_channels=eeg_channels
    )

    print("[INFO] Entering mainloop...")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[INFO] User interrupted.")
    finally:
        if board is not None:
            board.stop_stream()
            board.release_session()

if __name__ == '__main__':
    main()
