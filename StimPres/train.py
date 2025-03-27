import os
import numpy as np
import pandas as pd
import traceback
from collections import Counter

from scipy.signal import welch, butter, lfilter, iirnotch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
import joblib

# ============= FLAGS & CONFIG =============

TRAIN_MODE_RANDOM_ONLY = False
TRY_ALL_MODELS = False            # If True, tries multiple classifiers per binary model

# Use Hybrid Random Classifier as default
SINGLE_MODEL = ("MLP", {"hidden_layer_sizes":(128,64), "max_iter":10000, "alpha":1e-10})
ADD_RANDOM_GUESS_CLASSIFIER = True

USE_BANDPOWER_FEATURES = True
USE_RAW_FLATTENING = False
N_AUG = 10
OVERSAMPLE_FACTOR = 2
MAX_ATTEMPTS = 40
BIAS_THRESHOLD = 0.05

# ---------- DATA / FILES -----------
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
SUBTRIAL_SAMPLES = 250
GROUP_SIZE = 4

# Frequency bands for bandpower
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

# All classifier parameter sets (for TRY_ALL_MODELS mode)
CLASSIFIER_PARAM_SETS = [
    ("SVM", {"C": 1.0,  "gamma": "scale", "class_weight": None}),
    ("SVM", {"C": 1.0,  "gamma": "scale", "class_weight": "balanced"}),
    ("SVM", {"C": 10.0, "gamma": 0.1,     "class_weight": "balanced"}),
    ("RF",  {"n_estimators": 100, "max_depth": None, "class_weight": None}),
    ("RF",  {"n_estimators": 200, "max_depth": 10,   "class_weight": "balanced_subsample"}),
    ("LR",  {"C": 1.0, "class_weight": None}),
    ("LR",  {"C": 0.1, "class_weight": "balanced"}),
    ("MLP", {"hidden_layer_sizes": (64,),    "max_iter": 500, "alpha": 1e-3}),
    ("MLP", {"hidden_layer_sizes": (128, 64), "max_iter": 500, "alpha": 1e-4}),
]
if ADD_RANDOM_GUESS_CLASSIFIER:
    CLASSIFIER_PARAM_SETS.append(("DUMMY", {"strategy": "uniform"}))
    CLASSIFIER_PARAM_SETS.append(("DUMMY", {"strategy": "stratified"}))
# Also add the hashed random approach:
CLASSIFIER_PARAM_SETS.append(("HASHED_RANDOM", {"n_classes": NUM_CLASSES, "stable": True}))

# ------------------ CUSTOM CLASSIFIERS ------------------

class HybridRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, alpha=0.5, n_classes=NUM_CLASSES, stable=False):
        """
        If stable is False, an ephemeral component is added so that each prediction
        can vary even on similar inputs.
        """
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.n_classes = n_classes
        self.stable = stable

    def fit(self, X, y):
        from sklearn.utils.validation import check_X_y
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
        from sklearn.utils.validation import check_is_fitted, check_array
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        # Get base estimator probabilities.
        base_proba = self.base_estimator.predict_proba(X)  # shape: (n_samples, n_classes)
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        for i in range(n_samples):
            row = X[i]
            # Use the row's bytes as seed.
            row_bytes = row.tobytes()
            hash_val = hash(row_bytes)
            if self.stable:
                seed = hash_val % (2**32)
            else:
                # Incorporate an ephemeral random element.
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
    def __init__(self, n_classes=NUM_CLASSES, stable=True):
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

# ------------------ FILTERS & FEATURE EXTRACTION ------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(sig, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, sig)

def apply_notch(sig, freq, fs, q=30.0):
    nyq = fs / 2
    f0 = freq / nyq
    b, a = iirnotch(f0, q)
    return lfilter(b, a, sig)

def artifact_removal(eeg_array, threshold=150e-6):
    for ch in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch, :]) > threshold):
            eeg_array[ch, :] = 0.0
    return eeg_array

def compute_band_power(sig, fs, band):
    fmin, fmax = band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    return np.mean(psd[idx]) if len(idx) > 0 else 0.0

def extract_bandpower(eeg_array):
    feats = []
    for ch in range(eeg_array.shape[0]):
        sig = eeg_array[ch, :]
        for (lo, hi) in BANDS.values():
            feats.append(compute_band_power(sig, SAMPLE_RATE, (lo, hi)))
    return np.array(feats)

def extract_raw_flatten(eeg_array):
    return eeg_array.flatten()

def get_features(eeg_array):
    feat_list = []
    if USE_BANDPOWER_FEATURES:
        feat_list.append(extract_bandpower(eeg_array))
    if USE_RAW_FLATTENING:
        feat_list.append(extract_raw_flatten(eeg_array))
    if not feat_list:
        return np.zeros((1,))
    if len(feat_list) == 1:
        return feat_list[0]
    else:
        return np.concatenate(feat_list)

# ------------------ DATA AUGMENTATION & PARSING ------------------

def augment_trial(big_trial_array):
    out = []
    original = big_trial_array.copy()
    for i_aug in range(N_AUG):
        arr = original.copy()
        shift = np.random.randint(-30, 31)
        if shift > 0:
            arr = np.pad(arr, ((0, 0), (shift, 0)), mode='constant')[:, :arr.shape[1]]
        elif shift < 0:
            arr = np.pad(arr, ((0, 0), (0, -shift)), mode='constant')[:, :arr.shape[1]]
        scale = np.random.uniform(0.5, 1.5)
        arr *= scale
        noise = np.random.normal(0, 1e-4, arr.shape)
        arr += noise
        for ch in range(arr.shape[0]):
            if np.random.rand() < 0.3:
                arr[ch, :] = -arr[ch, :]
        if np.random.rand() < 0.3:
            c1, c2 = np.random.choice(arr.shape[0], 2, replace=False)
            tmp = arr[c1, :].copy()
            arr[c1, :] = arr[c2, :]
            arr[c2, :] = tmp
        out.append(arr)
    return out

def parse_file_group_4(filepath, label):
    print(f"[INFO] Parsing file => {filepath} with label={label}")
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return np.zeros((0, 64)), np.zeros((0,))
    try:
        df = pd.read_csv(filepath, sep=r'\s+|,|\t', engine='python', header=None)
    except Exception as e:
        print(f"[ERROR] Failed reading {filepath}: {e}")
        return np.zeros((0, 64)), np.zeros((0,))
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < (1 + NUM_CHANNELS):
        print(f"[ERROR] Not enough columns in {filepath} => {df.shape[1]}, skipping.")
        return np.zeros((0, 64)), np.zeros((0,))
    arr = df.values
    small_trials = []
    current = []
    def finalize_subtrial(rows):
        if not rows:
            return None
        raw = np.stack(rows, axis=-1)
        ns = raw.shape[1]
        if ns < SUBTRIAL_SAMPLES:
            pad = SUBTRIAL_SAMPLES - ns
            raw = np.pad(raw, ((0, 0), (0, pad)), mode='constant')
        elif ns > SUBTRIAL_SAMPLES:
            raw = raw[:, :SUBTRIAL_SAMPLES]
        return raw
    for i in range(arr.shape[0]):
        row = arr[i]
        sample_num = row[0]
        eegvals = row[1:1+NUM_CHANNELS]
        if sample_num == 0 and i != 0:
            st = finalize_subtrial(current)
            if st is not None:
                small_trials.append(st)
            current = []
        current.append(eegvals)
    if current:
        st = finalize_subtrial(current)
        if st is not None:
            small_trials.append(st)
    print(f"[DEBUG] Found {len(small_trials)} small sub-trials in file => {filepath}")
    big_trials = []
    for start_idx in range(0, len(small_trials), GROUP_SIZE):
        chunk = small_trials[start_idx: start_idx+GROUP_SIZE]
        big_arr = np.concatenate(chunk, axis=1)
        big_trials.append(big_arr)
    feats_list = []
    labels_list = []
    for big in big_trials:
        for ch in range(NUM_CHANNELS):
            big[ch, :] = apply_bandpass(big[ch, :], 1.0, 50.0, SAMPLE_RATE)
            big[ch, :] = apply_notch(big[ch, :], 60.0, SAMPLE_RATE, 30.0)
        big = artifact_removal(big)
        feat = get_features(big)
        feats_list.append(feat)
        labels_list.append(label)
        augs = augment_trial(big)
        for ag in augs:
            feats_list.append(get_features(ag))
            labels_list.append(label)
    X = np.array(feats_list)
    y = np.array(labels_list)
    print(f"[DEBUG] parse_file_group_4 => returning shapes X={X.shape}, y={y.shape}")
    return X, y

def oversample(X, y, factor=2):
    print(f"[INFO] Oversampling with factor={factor}")
    Xout = []
    yout = []
    for i in range(len(X)):
        Xout.append(X[i])
        yout.append(y[i])
        for _ in range(factor - 1):
            Xout.append(X[i])
            yout.append(y[i])
    Xo = np.array(Xout)
    yo = np.array(yout)
    print(f"[DEBUG] Oversample => new shape X={Xo.shape}, y={yo.shape}")
    return Xo, yo

def measure_bias(y_true, y_pred):
    c = Counter(y_pred)
    top_count = max(c.values()) if c else 0
    total = len(y_pred)
    return top_count / total if total > 0 else 1.0

def create_classifier(name, params):
    print(f"[DEBUG] create_classifier => name={name}, params={params}")
    if name == "SVM":
        from sklearn.svm import SVC
        c = params.get("C", 1.0)
        g = params.get("gamma", "scale")
        cw = params.get("class_weight", None)
        clf = SVC(kernel='rbf', probability=True, C=c, gamma=g, class_weight=cw, random_state=None)
        return clf
    elif name == "HYBRID_RANDOM":
        alpha = params.get("alpha", 0.5)
        stable = params.get("stable", False)
        from sklearn.linear_model import LogisticRegression
        base_est = LogisticRegression(max_iter=1000)
        return HybridRandomClassifier(base_estimator=base_est, alpha=alpha, n_classes=params.get("n_classes", NUM_CLASSES), stable=stable)
    elif name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        ne = params.get("n_estimators", 100)
        md = params.get("max_depth", None)
        cw = params.get("class_weight", None)
        clf = RandomForestClassifier(n_estimators=ne, max_depth=md, class_weight=cw, random_state=None)
        return clf
    elif name == "LR":
        from sklearn.linear_model import LogisticRegression
        c = params.get("C", 1.0)
        cw = params.get("class_weight", None)
        clf = LogisticRegression(C=c, class_weight=cw, max_iter=1000, random_state=None)
        return clf
    elif name == "MLP":
        from sklearn.neural_network import MLPClassifier
        hls = params.get("hidden_layer_sizes", (64,))
        alpha = params.get("alpha", 1e-4)
        mi = params.get("max_iter", 500)
        clf = MLPClassifier(hidden_layer_sizes=hls, alpha=alpha, max_iter=mi, random_state=None)
        return clf
    elif name == "DUMMY":
        strategy = params.get("strategy", "uniform")
        clf = DummyClassifier(strategy=strategy, random_state=None)
        return clf
    elif name == "HASHED_RANDOM":
        n_cls = params.get("n_classes", NUM_CLASSES)
        stable = params.get("stable", True)
        clf = HashedRandomClassifier(n_classes=n_cls, stable=stable)
        return clf
    else:
        from sklearn.svm import SVC
        print(f"[WARN] Unknown name={name}, defaulting to SVC(probability=True)")
        return SVC(probability=True, random_state=None)

def main():
    try:
        print("[INFO] Starting main() of EEG training script.")
        
        if TRAIN_MODE_RANDOM_ONLY:
            print("[INFO] TRAIN_MODE_RANDOM_ONLY=True => Creating a purely random (uniform) model ignoring data.")
            X_fake = np.zeros((NUM_CLASSES, 64))
            y_fake = np.arange(NUM_CLASSES)
            random_model = DummyClassifier(strategy="uniform", random_state=None)
            random_model.fit(X_fake, y_fake)
            joblib.dump(random_model, "eeg_image_classifier.joblib")
            print("[INFO] A random DummyClassifier is saved as 'eeg_image_classifier.joblib'. Exiting.")
            return
        
        allX = []
        allY = []
        for fname, lbl in FILE_LABELS:
            path = os.path.join(DATA_FOLDER, fname)
            X_, y_ = parse_file_group_4(path, lbl)
            if X_.shape[0] > 0:
                allX.append(X_)
                allY.append(y_)
            else:
                print(f"[WARN] No data from file={fname}, skipping.")
        if len(allX) == 0:
            print("[ERROR] No data loaded from any file! Exiting.")
            return
        
        X = np.concatenate(allX, axis=0)
        y = np.concatenate(allY, axis=0)
        print(f"[INFO] After concatenation => X.shape={X.shape}, y.shape={y.shape}")
        
        X, y = oversample(X, y, factor=OVERSAMPLE_FACTOR)
        X, y = shuffle(X, y, random_state=None)
        print("[INFO] shape after shuffle =>", X.shape, y.shape)
        
        print("[INFO] Fitting StandardScaler on entire data (pre-split).")
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.joblib")
        print("[INFO] Saved scaler as scaler.joblib.")
        
        # Create directory for models
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Train a separate binary classifier for each class
        for class_label in range(NUM_CLASSES):
            print(f"\n[INFO] Training binary classifier for class {class_label}...")
            # Binary labels: 1 if the sample is of class class_label, else 0.
            y_binary = (y == class_label).astype(np.int32)
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y_binary, test_size=0.2, random_state=None, stratify=y_binary
            )
            if TRY_ALL_MODELS:
                best_bias = 999.0
                best_clf_name = None
                best_clf_params = None
                attempts = 0
                from sklearn.model_selection import RepeatedStratifiedKFold
                rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=None)
                while attempts < MAX_ATTEMPTS:
                    idx = attempts % len(CLASSIFIER_PARAM_SETS)
                    clf_name, param_dict = CLASSIFIER_PARAM_SETS[idx]
                    attempts += 1
                    # For binary classification, override n_classes if applicable.
                    if clf_name in ["HASHED_RANDOM", "HYBRID_RANDOM"]:
                        param_dict["n_classes"] = 2
                    print(f"[INFO] Attempt {attempts}: classifier={clf_name}, params={param_dict}")
                    clf = create_classifier(clf_name, param_dict)
                    biases = []
                    fold_count = 0
                    for train_idx, test_idx in rskf.split(X_train, y_train):
                        fold_count += 1
                        X_trainCV = X_train[train_idx]
                        y_trainCV = y_train[train_idx]
                        X_testCV = X_train[test_idx]
                        y_testCV = y_train[test_idx]
                        clf.fit(X_trainCV, y_trainCV)
                        y_predCV = clf.predict(X_testCV)
                        b = measure_bias(y_testCV, y_predCV)
                        biases.append(b)
                        print(f"    Fold {fold_count}, bias = {b:.3f}")
                    mean_b = np.mean(biases)
                    print(f"[INFO] Mean bias = {mean_b:.3f}")
                    if mean_b < best_bias:
                        best_bias = mean_b
                        best_clf_name = clf_name
                        best_clf_params = param_dict
                    if mean_b <= BIAS_THRESHOLD:
                        print(f"[INFO] Bias threshold reached.")
                        break
                print(f"[INFO] Best classifier for class {class_label}: {best_clf_name} with params {best_clf_params}, bias={best_bias:.3f}")
                final_clf = create_classifier(best_clf_name, best_clf_params)
                final_clf.fit(X_train, y_train)
                print("[INFO] Evaluation on hold-out test set:")
                y_pred_test = final_clf.predict(X_test)
                print(classification_report(y_test, y_pred_test))
                final_clf.fit(Xs, y_binary)
            else:
                clf_name, param_dict = SINGLE_MODEL
                final_clf = create_classifier(clf_name, param_dict)
                final_clf.fit(X_train, y_train)
                print(f"[INFO] Evaluation for class {class_label} using {clf_name}:")
                y_pred_test = final_clf.predict(X_test)
                print(classification_report(y_test, y_pred_test))
                final_clf.fit(Xs, y_binary)
            
            out_path = os.path.join(models_dir, f"eeg_image_classifier_class_{class_label}.joblib")
            joblib.dump(final_clf, out_path)
            print(f"[INFO] Saved binary classifier for class {class_label} to {out_path}")
        
        print("[INFO] Done training all binary classifiers.")
    
    except Exception as e:
        print("[ERROR] in main()")
        traceback.print_exc()

if __name__ == '__main__':
    main()
