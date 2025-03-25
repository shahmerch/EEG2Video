import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler

# ----------------------
# LOAD MODEL & SCALER
# ----------------------
MODEL_PATH = "eeg_image_classifier.joblib"
SCALER_PATH = "scaler.joblib"
FIXED_SCALER_PATH = "scaler_fixed.joblib"

print("[INFO] Loading model from", MODEL_PATH, "...")
model = joblib.load(MODEL_PATH)

# Try loading original scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Scaler loaded from", SCALER_PATH)
except Exception as e:
    print("[WARNING] No scaler found, using unscaled features. Error:", e)
    scaler = None

# Try loading fixed scaler
try:
    fixed_scaler = joblib.load(FIXED_SCALER_PATH)
    print("[INFO] Fixed Scaler loaded from", FIXED_SCALER_PATH)
except Exception as e:
    print("[WARNING] No fixed scaler found. Ensure that it's trained on real EEG data.")
    fixed_scaler = None

# ----------------------
# FUNCTION: GENERATE RANDOM EEG DATA
# ----------------------
def generate_random_eeg():
    """
    Generates a random 16-channel EEG segment (256 samples per channel).
    """
    eeg_data = np.random.uniform(-500000, 500000, (16, 256))  # Simulated EEG in microvolts
    return eeg_data

# ----------------------
# FUNCTION: EXTRACT BAND-POWER FEATURES
# ----------------------
def extract_bandpower(eeg_data):
    """
    Computes bandpower features for each EEG channel.
    """
    feats = np.mean(np.abs(eeg_data), axis=1)  # Simplified feature extraction (e.g., mean absolute amplitude)
    return feats.reshape(1, -1)  # Ensure correct shape for classification

# ----------------------
# FUNCTION: TEST MODEL OUTPUT VARIATION
# ----------------------
def test_model(scaler_to_use, label="Default Scaler"):
    """
    Generates EEG data, extracts features, applies the given scaler (if available),
    and tests the classifier.
    """
    print(f"\n[INFO] Running test with {label} ...")
    
    # Generate EEG data
    raw_eeg = generate_random_eeg()

    # Extract features
    raw_feats = extract_bandpower(raw_eeg)
    print(f"[DEBUG] Raw Features (Before Scaling): mean={raw_feats.mean():.2f}, std={raw_feats.std():.2f}")

    # Scale features if a scaler is available
    if scaler_to_use:
        scaled_feats = scaler_to_use.transform(raw_feats)
        print(f"[DEBUG] Scaled Features: mean={scaled_feats.mean():.2f}, std={scaled_feats.std():.2f}")
    else:
        scaled_feats = raw_feats

    # Get probability outputs
    probs = model.predict_proba(scaled_feats)[0]
    print("[INFO] Probabilities:", probs)

    # Get raw decision function output (if available)
    try:
        decision_output = model.decision_function(scaled_feats)
        print("[DEBUG] Decision Function Output:", decision_output)
    except AttributeError:
        print("[WARNING] Model does not support decision_function()")

    return probs

# ----------------------
# MAIN TEST LOOP
# ----------------------
if __name__ == "__main__":
    print("\n[INFO] Model Details:")
    print("   Type:", type(model).__name__)
    print("   Parameters:", model.get_params())

    iterations = 5

    for i in range(1, iterations + 1):
        print(f"\n[INFO] Iteration {i}/{iterations} - generating EEG and classifying ...")
        test_model(scaler, label="Original Scaler")

        # Test without any scaling
        test_model(None, label="No Scaler")

        # If a fixed scaler is available, test with it
        if fixed_scaler:
            test_model(fixed_scaler, label="Fixed Scaler")

        time.sleep(1)

    print("\n[INFO] Done.")
