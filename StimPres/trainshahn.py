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
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
# ============= FLAGS & CONFIG =============

# If True, we skip real training & produce a basic random model (DummyClassifier).
# (You could also adapt this to produce "HashedRandomClassifier" if you want.)
TRAIN_MODE_RANDOM_ONLY = False

TRY_ALL_MODELS = False            # If True, tries multiple classifiers
SINGLE_MODEL =    ("HYBRID_RANDOM", {"alpha":1.0, "stable":True})

ADD_RANDOM_GUESS_CLASSIFIER = True

# We'll also add "HASHED_RANDOM" to param sets below
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
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta":  (13,30),
}

# All classifier param sets
CLASSIFIER_PARAM_SETS = [
    ("SVM", {"C":1.0,  "gamma":"scale", "class_weight":None}),
    ("SVM", {"C":1.0,  "gamma":"scale", "class_weight":"balanced"}),
    ("SVM", {"C":10.0, "gamma":0.1,     "class_weight":"balanced"}),
    ("RF",  {"n_estimators":100, "max_depth":None, "class_weight":None}),
    ("RF",  {"n_estimators":200, "max_depth":10,   "class_weight":"balanced_subsample"}),
    ("LR",  {"C":1.0, "class_weight":None}),
    ("LR",  {"C":0.1, "class_weight":"balanced"}),
    ("MLP", {"hidden_layer_sizes":(64,),    "max_iter":500, "alpha":1e-3}),
    ("MLP", {"hidden_layer_sizes":(128,64), "max_iter":500, "alpha":1e-4}),
    # You can remove these if you don't want the regular dummy classifiers:
]
if ADD_RANDOM_GUESS_CLASSIFIER:
    CLASSIFIER_PARAM_SETS.append(("DUMMY", {"strategy": "uniform"}))
    CLASSIFIER_PARAM_SETS.append(("DUMMY", {"strategy": "stratified"}))

# Add the hashed random approach:
CLASSIFIER_PARAM_SETS.append(("HASHED_RANDOM", {"n_classes": 12, "stable": True}))

class HybridRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, alpha=0.5, n_classes=12, stable=True):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.n_classes = n_classes
        self.stable = stable

    def fit(self, X, y):
        """
        Fit the base estimator on the training data. Must have n_classes unique labels in y.
        """
        X, y = check_X_y(X, y)
        # We assume the classes are 0..(n_classes-1). If not, we might handle that in a LabelEncoder.
        if self.base_estimator is None:
            # default to a logistic regression, or any other method
            from sklearn.linear_model import LogisticRegression
            self.base_estimator = LogisticRegression(max_iter=1000)

        self.base_estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict the class with highest final probability from predict_proba(X).
        """
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        """
        1) get the base estimator's predicted probabilities
        2) combine them with a hashed-random distribution
        3) re-normalize => final probability
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        base_proba = self.base_estimator.predict_proba(X)  # shape => (n_samples, n_classes)
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        alpha = self.alpha

        for i in range(n_samples):
            # get random distribution for row i
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
            randvals /= randvals.sum()  # shape => (n_classes, )

            # Weighted blend
            # base_proba[i] is (n_classes,) 
            blended = alpha * base_proba[i] + (1 - alpha) * randvals
            # re-normalize just to be sure
            blended /= blended.sum()

            out[i] = blended

        return out

# ------------------------------------------------------------------------------------
#                           CUSTOM HASHED RANDOM CLASSIFIER
# ------------------------------------------------------------------------------------
class HashedRandomClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that outputs unique random probabilities for each input row X,
    using the row values as a seed.

    - n_classes: number of output classes
    - stable: if True, same input => same seed => same probability distribution
              if False, we add ephemeral randomness, so it changes each call
    """
    def __init__(self, n_classes=12, stable=True):
        self.n_classes = n_classes
        self.stable = stable

    def fit(self, X, y=None):
        # No real training, but we might store n_classes from y if we want
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        for i in range(n_samples):
            row = X[i]
            # We'll convert row to bytes and hash it
            row_bytes = row.tobytes()
            hash_val = hash(row_bytes)

            if self.stable:
                # same input => same seed => same distribution
                np.random.seed(hash_val % (2**32))
            else:
                # incorporate ephemeral randomness
                # e.g. combine row hash with a random int
                ephemeral = np.random.randint(0, 2**31)
                combined_seed = (hash_val ^ ephemeral) % (2**32)
                np.random.seed(combined_seed)

            randvals = np.random.rand(self.n_classes)
            randvals /= randvals.sum()
            out[i] = randvals

        return out




# ---------- FILTER & ARTIFACT ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    from scipy.signal import butter
    b,a = butter(order, [low, high], btype='band')
    return b,a

def apply_bandpass(sig, lowcut, highcut, fs):
    b,a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b,a,sig)

def apply_notch(sig, freq, fs, q=30.0):
    nyq=fs/2
    f0=freq/nyq
    b,a = iirnotch(f0,q)
    return lfilter(b,a,sig)

def artifact_removal(eeg_array, threshold=150e-6):
    for ch in range(eeg_array.shape[0]):
        if np.any(np.abs(eeg_array[ch,:])>threshold):
            eeg_array[ch,:]=0.0
    return eeg_array

# ---------- Feature extraction ----------
def compute_band_power(sig, fs, band):
    fmin,fmax=band
    freqs, psd = welch(sig, fs=fs, nperseg=128)
    idx = np.where((freqs>=fmin)&(freqs<=fmax))[0]
    return np.mean(psd[idx]) if len(idx)>0 else 0.0

def extract_bandpower(eeg_array):
    feats=[]
    for ch in range(eeg_array.shape[0]):
        sig=eeg_array[ch,:]
        for (lo,hi) in BANDS.values():
            val = compute_band_power(sig, SAMPLE_RATE, (lo,hi))
            feats.append(val)
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

# ---------- Data augmentation ----------
def augment_trial(big_trial_array):
    out=[]
    original = big_trial_array.copy()
    for i_aug in range(N_AUG):
        arr = original.copy()
        shift = np.random.randint(-30,31)
        if shift>0:
            arr = np.pad(arr, ((0,0),(shift,0)), mode='constant')[:,:arr.shape[1]]
        elif shift<0:
            arr = np.pad(arr, ((0,0),(0,-shift)), mode='constant')[:,:arr.shape[1]]

        scale = np.random.uniform(0.5,1.5)
        arr *= scale

        noise = np.random.normal(0,1e-4,arr.shape)
        arr += noise

        for ch in range(arr.shape[0]):
            if np.random.rand()<0.3:
                arr[ch,:] = -arr[ch,:]

        if np.random.rand()<0.3:
            c1,c2 = np.random.choice(arr.shape[0],2,replace=False)
            tmp = arr[c1,:].copy()
            arr[c1,:] = arr[c2,:]
            arr[c2,:] = tmp

        out.append(arr)
    return out

# ---------- parse_file_group_4 ----------
def parse_file_group_4(filepath, label):
    print(f"[INFO] Parsing file => {filepath} with label={label}")
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return np.zeros((0,64)), np.zeros((0,))

    try:
        df = pd.read_csv(filepath, sep=r'\s+|,|\t', engine='python', header=None)
    except Exception as e:
        print(f"[ERROR] Failed reading {filepath}: {e}")
        return np.zeros((0,64)), np.zeros((0,))
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < (1+NUM_CHANNELS):
        print(f"[ERROR] Not enough columns in {filepath} => {df.shape[1]}, skipping.")
        return np.zeros((0,64)), np.zeros((0,))

    arr = df.values
    small_trials=[]
    current=[]

    def finalize_subtrial(rows):
        if not rows:
            return None
        raw = np.stack(rows, axis=-1)
        ns = raw.shape[1]
        if ns<SUBTRIAL_SAMPLES:
            pad = SUBTRIAL_SAMPLES - ns
            raw = np.pad(raw, ((0,0),(0,pad)), mode='constant')
        elif ns>SUBTRIAL_SAMPLES:
            raw = raw[:,:SUBTRIAL_SAMPLES]
        return raw

    for i in range(arr.shape[0]):
        row = arr[i]
        sample_num = row[0]
        eegvals = row[1:1+NUM_CHANNELS]
        if sample_num==0 and i!=0:
            st = finalize_subtrial(current)
            if st is not None:
                small_trials.append(st)
            current=[]
        current.append(eegvals)
    if current:
        st = finalize_subtrial(current)
        if st is not None:
            small_trials.append(st)

    print(f"[DEBUG] Found {len(small_trials)} small sub-trials in file => {filepath}")

    big_trials=[]
    for start_idx in range(0, len(small_trials), GROUP_SIZE):
        chunk = small_trials[start_idx: start_idx+GROUP_SIZE]
        # if len(chunk)<GROUP_SIZE:
        #     print(f"[WARN] Skipping incomplete group => needed {GROUP_SIZE}, got {len(chunk)}")
        #     break
        big_arr = np.concatenate(chunk, axis=1)
        big_trials.append(big_arr)

    feats_list=[]
    labels_list=[]
    for big in big_trials:
        for ch in range(NUM_CHANNELS):
            big[ch,:] = apply_bandpass(big[ch,:], 1.0,50.0, SAMPLE_RATE)
            big[ch,:] = apply_notch(big[ch,:], 60.0, SAMPLE_RATE, 30.0)
        big = artifact_removal(big)

        feat = get_features(big)
        feats_list.append(feat)
        labels_list.append(label)

        augs = augment_trial(big)
        for ag in augs:
            f2 = get_features(ag)
            feats_list.append(f2)
            labels_list.append(label)

    X = np.array(feats_list)
    y = np.array(labels_list)
    print(f"[DEBUG] parse_file_group_4 => returning shapes X={X.shape}, y={y.shape}")
    return X,y

# ---------- Oversample -----------
def oversample(X, y, factor=2):
    print(f"[INFO] Oversampling with factor={factor}")
    Xout=[]
    yout=[]
    for i in range(len(X)):
        Xout.append(X[i])
        yout.append(y[i])
        for _ in range(factor-1):
            Xout.append(X[i])
            yout.append(y[i])
    Xo = np.array(Xout)
    yo = np.array(yout)
    print(f"[DEBUG] Oversample => new shape X={Xo.shape}, y={yo.shape}")
    return Xo, yo

# ---------- measure bias -----------
def measure_bias(y_true, y_pred):
    c=Counter(y_pred)
    top_count=max(c.values()) if c else 0
    total=len(y_pred)
    return top_count/total if total>0 else 1.0

def create_classifier(name, params):
    print(f"[DEBUG] create_classifier => name={name}, params={params}")

    if name=="SVM":
        from sklearn.svm import SVC
        c=params.get("C",1.0)
        g=params.get("gamma","scale")
        cw=params.get("class_weight",None)
        clf=SVC(kernel='rbf', probability=True, C=c, gamma=g, class_weight=cw, random_state=None)
        return clf
    elif name == "HYBRID_RANDOM":
        # parse out alpha, stable, etc.
        alpha = params.get("alpha", 0.5)
        stable = params.get("stable", True)
        # you can specify a base_estimator inside params if desired
        from sklearn.linear_model import LogisticRegression
        base_est = LogisticRegression(max_iter=1000)
        return HybridRandomClassifier(
            base_estimator=base_est,
            alpha=alpha,
            n_classes=12,  # or however many classes
            stable=stable
    )

    elif name=="RF":
        from sklearn.ensemble import RandomForestClassifier
        ne=params.get("n_estimators",100)
        md=params.get("max_depth",None)
        cw=params.get("class_weight",None)
        clf=RandomForestClassifier(n_estimators=ne, max_depth=md, class_weight=cw, random_state=None)
        return clf

    elif name=="LR":
        from sklearn.linear_model import LogisticRegression
        c=params.get("C",1.0)
        cw=params.get("class_weight",None)
        clf=LogisticRegression(C=c,class_weight=cw,max_iter=1000,random_state=None)
        return clf

    elif name=="MLP":
        from sklearn.neural_network import MLPClassifier
        hls=params.get("hidden_layer_sizes",(64,))
        alpha=params.get("alpha",1e-4)
        mi=params.get("max_iter",500)
        clf=MLPClassifier(hidden_layer_sizes=hls, alpha=alpha, max_iter=mi, random_state=None)
        return clf

    elif name=="DUMMY":
        strategy=params.get("strategy","uniform")
        clf=DummyClassifier(strategy=strategy, random_state=None)
        return clf

    elif name=="HASHED_RANDOM":
        # parse out 'n_classes' or 'stable'
        n_classes = params.get("n_classes", 12)
        stable = params.get("stable", True)
        clf = HashedRandomClassifier(n_classes=n_classes, stable=stable)
        return clf

    else:
        from sklearn.svm import SVC
        print(f"[WARN] Unknown name={name}, defaulting to SVC(probability=True)")
        return SVC(probability=True, random_state=None)

def main():
    try:
        print("[INFO] Starting main() of EEG training script.")
        
        # --------------- RANDOM ONLY TRAIN MODE ---------------
        if TRAIN_MODE_RANDOM_ONLY:
            print("[INFO] TRAIN_MODE_RANDOM_ONLY=True => Creating a purely random (uniform) model ignoring data.")
            
            # We'll produce the dummy with correct # classes => 12
            X_fake = np.zeros((12,64))  # 12 rows, 64 features
            y_fake = np.arange(12)      # 0..11
            random_model = DummyClassifier(strategy="uniform", random_state=None)
            random_model.fit(X_fake, y_fake)
            joblib.dump(random_model, "eeg_image_classifier.joblib")
            print("[INFO] A random DummyClassifier is saved as 'eeg_image_classifier.joblib'. Exiting.")
            return
        # ------------------------------------------------------

        allX=[]
        allY=[]

        # parse each file
        for fname,lbl in FILE_LABELS:
            path = os.path.join(DATA_FOLDER, fname)
            X_,y_ = parse_file_group_4(path, lbl)
            if X_.shape[0]>0:
                allX.append(X_)
                allY.append(y_)
            else:
                print(f"[WARN] No data from file={fname}, skipping.")
        if len(allX)==0:
            print("[ERROR] No data loaded from any file! Exiting.")
            return

        X=np.concatenate(allX, axis=0)
        y=np.concatenate(allY, axis=0)
        print(f"[INFO] After concatenation => X.shape={X.shape}, y.shape={y.shape}")

        # oversample
        X,y=oversample(X,y,factor=OVERSAMPLE_FACTOR)

        # shuffle data
        X, y = shuffle(X, y, random_state=None)
        print("[INFO] shape after shuffle =>", X.shape, y.shape)

        # scale
        print("[INFO] Creating StandardScaler and fitting on entire data (pre-split).")
        scaler=StandardScaler()
        Xs=scaler.fit_transform(X)

        # split
        print("[INFO] Splitting train/test => 80%/20%.")
        Xs_train, Xs_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.2, random_state=None, stratify=y
        )
        print("[INFO] splitted => train={}, test={}".format(len(y_train), len(y_test)))

        if TRY_ALL_MODELS:
            print("[INFO] Searching across CLASSIFIER_PARAM_SETS with repeated CV.")
            best_bias=999.0
            best_clf_name=None
            best_clf_params=None
            attempts=0

            from sklearn.model_selection import RepeatedStratifiedKFold
            rskf=RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=None)

            while attempts<MAX_ATTEMPTS:
                idx=attempts % len(CLASSIFIER_PARAM_SETS)
                clf_name, param_dict = CLASSIFIER_PARAM_SETS[idx]
                attempts+=1
                print(f"\n[INFO] attempt={attempts}, classifier={clf_name}, params={param_dict}")
                clf=create_classifier(clf_name, param_dict)

                biases=[]
                fold_count=0
                for train_idx, test_idx in rskf.split(Xs_train, y_train):
                    fold_count+=1
                    X_trainCV = Xs_train[train_idx]
                    y_trainCV = y_train[train_idx]
                    X_testCV  = Xs_train[test_idx]
                    y_testCV  = y_train[test_idx]
                    clf.fit(X_trainCV, y_trainCV)
                    y_predCV=clf.predict(X_testCV)
                    b=measure_bias(y_testCV, y_predCV)
                    biases.append(b)
                    print(f"    [DEBUG] Fold={fold_count}, bias={b:.3f}")
                mean_b=np.mean(biases)
                print(f"[INFO] => mean bias across folds = {mean_b:.3f}")

                if mean_b<best_bias:
                    best_bias=mean_b
                    best_clf_name=clf_name
                    best_clf_params=param_dict

                if mean_b<=BIAS_THRESHOLD:
                    print(f"[INFO] => success: bias <= {BIAS_THRESHOLD}, stopping search.")
                    break
                else:
                    print("[INFO] => still too biased, continuing search...")

            print(f"[INFO] Finished. best => classifier={best_clf_name}, params={best_clf_params}, bias={best_bias:.3f}")
            final_clf=create_classifier(best_clf_name,best_clf_params)
            final_clf.fit(Xs_train,y_train)

        else:
            clf_name, param_dict = SINGLE_MODEL
            print(f"[INFO] Single approach => {clf_name}, params={param_dict}")
            final_clf=create_classifier(clf_name, param_dict)
            final_clf.fit(Xs_train,y_train)

        # Evaluate
        print("[INFO] Predicting on hold-out set for final classification_report...")
        y_pred=final_clf.predict(Xs_test)
        print("\n[INFO] Final classification_report on hold-out =>")
        print(classification_report(y_test, y_pred))

        # retrain final on entire data
        print("[INFO] Retraining final model on entire dataset (Xs,y).")
        final_clf.fit(Xs,y)

        # save
        model_outpath = "eeg_image_classifier.joblib"
        scaler_outpath = "scaler.joblib"
        print(f"[INFO] Saving final model => {model_outpath}")
        joblib.dump(final_clf, model_outpath)
        print(f"[INFO] Saving scaler => {scaler_outpath}")
        joblib.dump(scaler, scaler_outpath)

        print("[INFO] Done. Exiting main() normally.")

    except Exception as e:
        print("[ERROR] in main()")
        traceback.print_exc()

if __name__=="__main__":
    main()
