# EEG Classification System

This repository contains a comprehensive EEG signal processing and classification system for brain-computer interface applications. The system consists of three main components: a stimulus presentation module (`StimPresInt.py`), a training module (`train.py`), and a real-time trial module (`trial.py`).

## Overview

The system processes EEG data to classify brain signals into one of 12 different classes (representing visual stimuli). It implements a complete pipeline from data collection to real-time classification:

- **Data Collection**: Synchronized stimulus presentation and EEG recording
- **Signal Processing**: Filtering, artifact removal, and feature extraction
- **Model Training**: Multi-class classification using binary classifiers
- **Real-time Classification**: Live processing and visualization of EEG signals

## System Architecture

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ StimPresInt.py│     │    train.py   │     │   trial.py    │
│   (Data       │────▶│  (Training    │────▶│  (Real-time   │
│  Collection)  │     │    Module)    │     │ Classification)│
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  EEG Data     │     │Trained Models │     │Classification │
│  Files (.txt) │     │& Scaler Files │     │  Results &    │
│               │     │               │     │ Visualization │
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## Environment Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (optional, for cloning the repository)

### macOS Setup

1. **Install Miniconda** (if not already installed):
   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
   bash miniconda.sh -b -p $HOME/miniconda
   source $HOME/miniconda/bin/activate
   ```

2. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Create and activate the conda environment**:
   ```bash
   conda create -n eeg-classify python=3.8
   conda activate eeg-classify
   ```

4. **Install required packages**:
   ```bash
   # Core scientific packages
   conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib joblib
   
   # Additional libraries
   conda install -c conda-forge pillow pyautogui keyboard
   
   # BrainFlow for OpenBCI interface
   pip install brainflow
   ```

5. **Install tkinter** (if not included with Python):
   ```bash
   conda install -c conda-forge python.app
   ```

6. **Verify the installation**:
   ```bash
   python -c "import numpy, scipy, pandas, sklearn, matplotlib, PIL, joblib, pyautogui, keyboard, brainflow; print('All packages imported successfully!')"
   ```

### Windows Setup

1. **Install Miniconda** (if not already installed):
   - Download the installer from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
   - Run the installer and follow the instructions

2. **Open Anaconda Prompt** (search for "Anaconda Prompt" in the Start menu)

3. **Clone or download the repository**:
   ```cmd
   git clone <repository-url>
   cd <repository-directory>
   ```
   Alternatively, download and extract the ZIP file from the repository website.

4. **Create and activate the conda environment**:
   ```cmd
   conda create -n eeg-classify python=3.8
   conda activate eeg-classify
   ```

5. **Install required packages**:
   ```cmd
   :: Core scientific packages
   conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib joblib
   
   :: Visualization and GUI
   conda install -c conda-forge pillow
   
   :: Additional libraries
   pip install pyautogui keyboard
   
   :: BrainFlow for OpenBCI interface
   pip install brainflow
   ```

6. **Verify the installation**:
   ```cmd
   python -c "import numpy, scipy, pandas, sklearn, matplotlib, PIL, joblib, pyautogui, keyboard, brainflow; print('All packages imported successfully!')"
   ```

### Troubleshooting

- **OpenBCI Board Connection Issues**:
  - On macOS, you may need to install additional drivers for the board: `brew install --cask silicon-labs-vcp-driver`
  - On Windows, install the [FTDI drivers](https://ftdichip.com/drivers/vcp-drivers/)

- **Tkinter Issues on macOS**:
  - If you encounter issues with tkinter when running trial.py, use `pythonw` instead of `python` to run GUI applications:
    ```bash
    pythonw trial.py
    ```

- **Package Conflicts**:
  - If you encounter conflicts, try creating the environment with fewer initial packages and add them one by one:
    ```bash
    conda create -n eeg-classify python=3.8
    conda activate eeg-classify
    conda install numpy
    # Add other packages incrementally
    ```

---

## Core Components

The system consists of three main Python scripts, each serving a specific purpose in the EEG classification pipeline.

---

## 1. StimPresInt.py - Stimulus Presentation & Data Collection

### Purpose
The stimulus presentation and data collection module responsible for displaying visual stimuli while synchronously recording EEG data from an OpenBCI board.

### Key Features
- **Synchronized Stimuli**: Presents PowerPoint slides with specific timing
- **EEG Recording**: Connects to OpenBCI Cyton+Daisy board (16 channels)
- **Session Control**: Enables pausing/resuming data collection with keyboard commands
- **Marker Insertion**: Places markers in EEG data to align with stimulus events
- **Data Storage**: Saves raw EEG data with structured naming convention

### Flow Diagram

```
┌─────────────────┐
│ Setup PowerPoint│
│   & OpenBCI     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  main_loop()    │◄───────────────┐
└────────┬────────┘                │
         │                         │
         ▼                         │
┌─────────────────┐                │
│ Select Random   │                │
│ Stimulus (0-11) │                │
└────────┬────────┘                │
         │                         │
         ▼                         │
┌─────────────────┐                │
│  one_block()    │                │
│ Display Stimulus│                │
└────────┬────────┘                │
         │                         │
         ▼                         │
┌─────────────────┐                │
│ Start EEG Stream│                │
└────────┬────────┘                │
         │                         │
         ▼                         │
┌─────────────────┐                │
│Recording Session│                │
│ (10 trials with │                │
│  markers)       │                │
└────────┬────────┘                │
         │                         │
         ▼                         │
┌─────────────────┐                │
│ Save EEG Data   │                │
│   to File       │                │
└────────┬────────┘                │
         │                         │
         ▼                         │
┌─────────────────┐      ┌─────────────────┐
│  More Tests     │─Yes─▶│ 2 Second Buffer │──┐
│  Remaining?     │      └─────────────────┘  │
└────────┬────────┘                           │
         │ No                                 │
         ▼                                    │
┌─────────────────┐                           │
│End Session & Exit│                          │
└─────────────────┘                           │
                                              │
                                              │
                    ┌─────────────────────────┘
                    │
                    ▼
```

### Key Variables and Types

| Variable | Type | Example | Description |
|----------|------|---------|-------------|
| `board` | `BoardShim` | `BoardShim(BoardIds.CYTON_DAISY_WIFI_BOARD, params)` | OpenBCI board interface |
| `params` | `BrainFlowInputParams` | `params.ip_address = "192.168.4.1"` | Configuration for board connection |
| `slides` | `list[list[str, int]]` | `[['boat_1', 2], ['dog_1', 2]]` | Stimulus names and counters |
| `num_tests` | `int` | `24` | Total number of test blocks to run |
| `pause_event` | `threading.Event` | `pause_event.set()` | Thread synchronization object |
| `data` | `numpy.ndarray` | Multidimensional EEG data | Raw board data with markers |

### Usage

```bash
python StimPresInt.py
```

This will:
1. Open the PowerPoint presentation specified in the script
2. Connect to the OpenBCI board
3. Guide you through the data collection process
4. Save EEG data files in the specified naming convention

---

## 2. train.py - Model Training & Feature Extraction

### Purpose
The training module responsible for processing raw EEG data files, extracting meaningful features, and training classifiers for each class.

### Key Features
- **Data Processing**: Loads EEG data from text files, applies filters, and extracts features
- **Feature Extraction**: Calculates band power in delta, theta, alpha, and beta frequency bands
- **Binary Classifiers**: Trains one classifier per class for a one-vs-all approach
- **Model Selection**: Can try multiple classifier types to find the best performing model
- **Data Augmentation**: Implements signal augmentation techniques to improve model robustness
- **Evaluation**: Provides classification reports for each model

### Processing Flow

```
┌─────────────────┐
│ Load EEG Files  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parse Files into│
│ Trials & Labels │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signal Processing │
│ (bandpass, notch,│
│ artifact removal)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Extraction│
│ (band power,    │
│ raw flattening) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Augmentation│
│ & Oversampling  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scale Features  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ For each class: │
└────────┬────────┘
         │
         ▼
┌─────────────────┐                  ┌─────────────────┐
│Create Binary    │                  │Try Multiple     │
│Classification   │◄─Yes──┤TRY_ALL_MODELS?├─No─┐
│Problem          │                  └─────────────────┘   │
└────────┬────────┘                                        │
         │                                                 │
         ▼                                                 │
┌─────────────────┐                                        │
│Train & Evaluate │                                        │
│Multiple Models  │                                        │
└────────┬────────┘                                        │
         │                                                 │
         ▼                                                 │
┌─────────────────┐                                        │
│Select Best Model│                                        │
│ (Lowest Bias)   │                                        │
└────────┬────────┘                                        │
         │                                                 │
         │                                                 │
         │       ┌────────────────────────────────────────┘
         │       │
         ▼       ▼
┌─────────────────────────────┐
│    Train Single Model       │
│    (RFC, SVM, etc.)         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Save Binary Classifier Model│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ More Classes to Process?    │─Yes─┐
└────────────┬────────────────┘     │
             │ No                   │
             ▼                      │
┌─────────────────────────────┐     │
│ Exit                        │     │
└─────────────────────────────┘     │
                                    │
                   ┌────────────────┘
                   │
                   ▼
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TRAIN_MODE_RANDOM_ONLY` | `bool` | `False` | Generate a purely random classifier |
| `TRY_ALL_MODELS` | `bool` | `False` | Try multiple classifier types to find best performer |
| `SINGLE_MODEL` | `tuple` | `("RF", {...})` | Default model type when not trying all models |
| `USE_BANDPOWER_FEATURES` | `bool` | `True` | Use frequency band power as features |
| `USE_RAW_FLATTENING` | `bool` | `False` | Use flattened raw signals as features |
| `N_AUG` | `int` | `10` | Number of augmented samples to generate per trial |
| `OVERSAMPLE_FACTOR` | `int` | `2` | Factor by which to oversample the data |

### Key Variables and Types

| Variable | Type | Example | Description |
|----------|------|---------|-------------|
| `BANDS` | `dict[str, tuple[int, int]]` | `{"delta": (1, 4), "theta": (4, 8)}` | Frequency bands for feature extraction |
| `X` | `numpy.ndarray` | `array([[0.1, 0.2, ...], ...])` | Feature matrix, shape (n_samples, n_features) |
| `y` | `numpy.ndarray` | `array([0, 1, 2, ...])` | Target labels, shape (n_samples,) |
| `allX` | `list[numpy.ndarray]` | List of feature matrices | Accumulated features from all files |
| `scaler` | `StandardScaler` | `StandardScaler()` | Feature normalization object |
| `final_clf` | Various | `RandomForestClassifier(n_estimators=200)` | Trained classifier object |

### Usage

```bash
python train.py
```

This will:
1. Load EEG data from the "All Nos" directory (containing trial subdirectories)
2. Process and extract features from each file
3. Train binary classifiers for each class
4. Save the models in the "models" directory
5. Save the feature scaler as "scaler.joblib"

---

## 3. trial.py - Real-time Classification

### Purpose
The real-time testing module for running and visualizing EEG classifications, supporting both hardware and simulation modes.

### Key Features
- **Real-time Processing**: Processes EEG data in real-time from hardware or simulation
- **Visualization**: Displays probability distribution across classes
- **Image Association**: Displays images associated with predicted classes
- **Signal Processing**: Applies the same filters and feature extraction as the training module
- **Simulation Mode**: Can run in simulation mode using pre-recorded data

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SIMULATOR_MODE` | `bool` | `True` | Run with simulated data instead of real hardware |
| `SIM_FILE` | `str` | `"./All Nos/Nos000/Nos000_7_3.txt"` | Path to the file containing simulation data |
| `USE_IMAGES` | `bool` | `True` | Display images associated with classes |
| `REFRESH_INTERVAL` | `int` | `200` | GUI refresh rate in milliseconds |

### Processing Flow

```
┌─────────────────┐
│ Load Models &   │
│ Scaler          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Initialize GUI   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│SIMULATOR_MODE?  │─No──┐
└────────┬────────┘     │
         │ Yes          │
         ▼              ▼
┌─────────────────┐   ┌─────────────────┐
│Load Simulation  │   │Connect to       │
│Data             │   │OpenBCI Board    │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
┌─────────────────────────────┐
│      Main Loop              │◄─────┐
└────────────┬────────────────┘      │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Acquire EEG Data (4 seconds)│       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Apply Signal Processing     │       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Extract Features            │       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Scale Features              │       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Get Predictions from Each   │       │
│ Binary Classifier           │       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Combine Predictions into    │       │
│ Multi-class Probabilities   │       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Update GUI Visualization    │       │
└────────────┬────────────────┘       │
             │                        │
             ▼                        │
┌─────────────────────────────┐       │
│ Wait for Refresh Interval   │──────┐│
└─────────────────────────────┘      ││
                                     ││
                                     ▼│
                          ┌──────────┘│
                          │           │
                          ▼           │
┌─────────────────────────────┐       │
│ Application Closed?         │─No────┘
└────────────┬────────────────┘
             │ Yes
             ▼
┌─────────────────────────────┐
│ Clean Up & Exit             │
└─────────────────────────────┘
```

### Key Variables and Types

| Variable | Type | Example | Description |
|----------|------|---------|-------------|
| `models` | `dict[int, BaseEstimator]` | `{0: RandomForestClassifier(), ...}` | Dictionary mapping class indices to binary classifiers |
| `scaler` | `StandardScaler` | `joblib.load("scaler.joblib")` | Loaded feature scaler |
| `board` | `BoardShim` | `BoardShim(BoardIds.CYTON_BOARD, params)` | OpenBCI board interface or None in simulator mode |
| `class_photo_images` | `list[ImageTk.PhotoImage]` | List of image objects | Images representing each class |
| `SIM_DATA` | `numpy.ndarray` | 2D array of simulated EEG data | Simulation data when not using real hardware |
| `probabilities` | `numpy.ndarray` | `array([0.1, 0.2, ...])` | Probability scores for each class |

### Usage

```bash
python trial.py
```

This will:
1. Load the trained models and scaler
2. Initialize the GUI
3. Begin processing EEG data (real or simulated)
4. Display real-time classification results

---

## Data Format and File Structure

### EEG Data Files

EEG data is stored in text files with the following format:
- Each row represents a single time point
- First column: sample number (integer)
- Columns 2-17: EEG channel values (float, typically in microvolts)

Example:
```
0 -10.23 15.67 2.34 ... -5.78
1 -11.45 14.32 3.21 ... -6.01
...
```

### File Naming Convention

```
[Subject ID]_[Class Number]_[Trial Number].txt
```

Example: `SM001_7_3.txt` represents:
- Subject ID: SM001
- Class Number: 7 (specific stimulus)
- Trial Number: 3 (third recording session)

### Directory Structure

```
/
├── All Nos/                  # Main data directory
│   ├── Nos000/               # Subject folder
│   │   ├── Nos000_0_3.txt    # EEG data files
│   │   ├── Nos000_1_3.txt
│   │   └── ...
│   ├── Nos001/               # Another subject
│   └── ...
├── models/                   # Trained classifier models
│   ├── eeg_image_classifier_class_0.joblib
│   ├── eeg_image_classifier_class_1.joblib
│   └── ...
├── stimulus.pptx             # PowerPoint for visual stimuli
├── scaler.joblib             # Saved feature scaler
├── StimPresInt.py            # Stimulus presentation script
├── train.py                  # Training script
└── trial.py                  # Real-time testing script
```

---

## Signal Processing Details

### Filtering

Both training and trial modules apply the same signal processing:

1. **Bandpass Filtering**: 1-50 Hz using 4th order Butterworth filter
   ```python
   def butter_bandpass(lowcut, highcut, fs, order=4):
       nyq = fs / 2
       low = lowcut / nyq
       high = highcut / nyq
       b, a = butter(order, [low, high], btype='band')
       return b, a
   ```

2. **Notch Filtering**: 60 Hz to remove power line interference
   ```python
   def apply_notch(sig, freq, fs, q=30.0):
       nyq = fs / 2
       f0 = freq / nyq
       b, a = iirnotch(f0, q)
       return lfilter(b, a, sig)
   ```

3. **Artifact Removal**: Zeroing out channels that exceed threshold
   ```python
   def artifact_removal(eeg_array, threshold=150e-6):
       for ch in range(eeg_array.shape[0]):
           if np.any(np.abs(eeg_array[ch, :]) > threshold):
               eeg_array[ch, :] = 0.0
       return eeg_array
   ```

### Feature Extraction

Features are extracted from the processed EEG data:

1. **Band Power Features**: Power in delta, theta, alpha, and beta bands
   ```python
   def compute_band_power(sig, fs, band):
       fmin, fmax = band
       freqs, psd = welch(sig, fs=fs, nperseg=128)
       idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
       return np.mean(psd[idx]) if len(idx) > 0 else 0.0
   ```

2. **Raw Flattening** (optional): Direct flattening of filtered signals
   ```python
   def extract_raw_flatten(eeg_array):
       return eeg_array.flatten()
   ```

---

## Custom Classifiers

The system includes two custom classifier implementations:

### HybridRandomClassifier

Combines a trained classifier with randomization:

```python
class HybridRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, alpha=0.5, n_classes=NUM_CLASSES, stable=False):
        self.base_estimator = base_estimator  # Underlying classifier (e.g., LogisticRegression)
        self.alpha = alpha  # Weight between actual prediction and randomness
        self.n_classes = n_classes  # Number of output classes
        self.stable = stable  # Whether to use deterministic random generator
```

This classifier is useful for applications where:
- Deterministic but seemingly random outputs are needed
- A blend between true prediction and randomization is desired
- Control over the randomness factor is required

### HashedRandomClassifier

Creates reproducible random classifications based on input hashing:

```python
class HashedRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=NUM_CLASSES, stable=True):
        self.n_classes = n_classes  # Number of output classes
        self.stable = stable  # Whether to use deterministic hashing
```

This classifier generates consistent outputs for identical inputs by:
1. Hashing the input feature vector
2. Using the hash as a seed for a random number generator
3. Generating a probability distribution over classes

It's particularly useful for applications where:
- Reproducibility is important
- The system needs to simulate prediction behavior
- Training data is insufficient for reliable predictions

## Code File Configuration Flags

### trial.py
This file contains the real-time EEG classification system with the following configuration flags:

- `SIMULATOR_MODE`: Boolean flag to enable/disable simulation mode
  - `True`: Simulates sub-trials using pre-recorded data
  - `False`: Connects to actual EEG board
- `USE_IMAGES`: Boolean flag to enable/disable image display
- `NUM_CLASSES`: Number of classification categories (default: 12)
- `SAMPLE_RATE`: EEG sampling rate in Hz (default: 250)
- `SUBTRIAL_SAMPLES`: Number of samples per sub-trial (default: 250)
- `GROUP_SIZE`: Size of trial groups (default: 4)
- `NUM_CHANNELS`: Number of EEG channels (default: 16)
- `REFRESH_INTERVAL`: UI refresh interval in milliseconds (default: 200)
- `BANDPASS_LOW`: Lower cutoff frequency for bandpass filter (default: 1.0 Hz)
- `BANDPASS_HIGH`: Upper cutoff frequency for bandpass filter (default: 50.0 Hz)
- `NOTCH_FREQ`: Notch filter frequency (default: 60.0 Hz)
- `NOTCH_Q`: Notch filter Q factor (default: 30.0)
- `ARTIFACT_THRESHOLD`: Threshold for artifact removal (default: 150e-6)
- `USE_BANDPOWER_FEATURES`: Boolean flag to enable/disable bandpower feature extraction
- `USE_RAW_FLATTENING`: Boolean flag to enable/disable raw signal flattening

### train.py
This file handles the training of EEG classifiers with the following configuration flags:

- `TRAIN_MODE_RANDOM_ONLY`: Boolean flag to enable random-only training mode
- `TRY_ALL_MODELS`: Boolean flag to test multiple classifiers per binary model
- `SINGLE_MODEL`: Tuple specifying default classifier type and parameters
- `ADD_RANDOM_GUESS_CLASSIFIER`: Boolean flag to include random guess classifier
- `USE_BANDPOWER_FEATURES`: Boolean flag to enable/disable bandpower feature extraction
- `USE_RAW_FLATTENING`: Boolean flag to enable/disable raw signal flattening
- `N_AUG`: Number of augmented samples per trial (default: 10)
- `OVERSAMPLE_FACTOR`: Factor for oversampling minority classes (default: 2)
- `MAX_ATTEMPTS`: Maximum attempts for bias correction (default: 40)
- `BIAS_THRESHOLD`: Threshold for acceptable class bias (default: 0.05)
- `NUM_CLASSES`: Number of classification categories (default: 12)
- `NUM_CHANNELS`: Number of EEG channels (default: 16)
- `SAMPLE_RATE`: EEG sampling rate in Hz (default: 250)
- `SUBTRIAL_SAMPLES`: Number of samples per sub-trial (default: 250)
- `GROUP_SIZE`: Size of trial groups (default: 4)

### StimPresInt.py
This file manages the stimulus presentation interface with the following configuration flags:

- `num_tests`: Total number of tests to run (default: 24)
- `wait_slide`: Slide number for wait screen (default: '15')
- `begin_test_slide`: Slide number for test start screen (default: '2')
- `blank_slide`: Slide number for blank screen (default: '16')
- `slides`: List of stimulus slides with presentation counts
- `test_slides`: List tracking test counts for each stimulus
- `daisy_channels`: Channel configuration for Daisy board
- Board Configuration:
  - Sample rate: 250 Hz (configured via "~6")
  - Marker mode: Enabled (configured via "/4")
  - Channel configuration: Set via "x" commands