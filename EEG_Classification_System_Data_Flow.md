# EEG Classification System: Complete Data Flow and Process Documentation

This document provides a comprehensive explanation of the EEG classification system, following the data flow from stimulus presentation and EEG data collection through processing, training, and real-time classification.

## 1. System Overview

The EEG classification system is designed to recognize and classify brain signals into one of 12 different classes associated with visual stimuli (images). The system implements a complete machine learning pipeline with three primary components:

1. **StimPresInt.py**: Data collection module that synchronizes visual stimulus presentation with EEG recording
2. **train.py**: Model training module that processes raw EEG data, extracts features, and trains binary classifiers
3. **trial.py**: Real-time classification module that processes incoming EEG signals and predicts the associated stimulus

## 2. Data Collection (StimPresInt.py)

### 2.1 Process Flow

The data collection process operates as follows:

1. **System Initialization**:
   - Configures the OpenBCI Cyton+Daisy board (16-channel EEG) with specific parameters
   - Sets up PowerPoint presentation for visual stimulus display
   - Creates a thread to monitor keyboard input for pause/resume functionality

2. **Stimulus Presentation Sequence**:
   - Opens PowerPoint presentation containing visual stimuli
   - Switches to full-screen mode and displays rules to subject
   - Random selection of stimuli from 12 options (boat, dog, apple, cat, bowling, banana - each with 2 variations)

3. **EEG Data Recording**:
   - For each stimulus, the `one_block()` function:
     - Displays the selected image for 5 seconds
     - Starts EEG stream recording
     - Conducts 10 subtrial recordings with synchronized markers
     - Each subtrial consists of a wait slide, blank slide, and 2-second recording period
     - Markers are inserted at the beginning and end of each 2-second recording
   - Data is saved with naming convention: `[SubjectID]_[StimulusIndex]_3.txt`

### 2.2 Key Variables and Parameters

- `board`: OpenBCI BoardShim object for hardware communication
- `slides`: List of [stimulus_name, counter] pairs for randomized presentation
- `num_tests`: Number of test blocks to run (default: 24)
- `wait_slide`, `blank_slide`: PowerPoint slide indices for protocol timing

### 2.3 Timing and Synchronization

- Each stimulus is shown for 5 seconds for subject analysis
- Each recording subtrial has a 1-second buffer followed by a 2-second recording window
- Markers are inserted at precise times to enable segmentation during processing
- The system can be paused/resumed with 'x' key for experimenter control

### 2.4 Output Format

The output data files are saved as text files with the following structure:
- Column 1: Sample number (integer)
- Columns 2-17: EEG channel values (16 channels, floating-point values in microvolts)
- Each file contains multiple subtrials (10 subtrials of 2-second recordings)
- Markers in the data indicate segment boundaries

## 3. Data Processing and Model Training (train.py)

### 3.1 Data Loading and Preprocessing

1. **File Loading and Organization**:
   - The system searches for EEG data files in the "All Nos" directory
   - The directory structure is: `All Nos/[SubjectFolder]/[SubjectID]_[ClassNumber]_[TrialNumber].txt`
   - Files are parsed and labeled based on their filenames

2. **File Parsing (parse_file_group_4)**:
   - Each file is segmented into "subtrials" based on sample number resets
   - Subtrials are grouped into "big trials" of GROUP_SIZE (4 subtrials each)
   - Each big trial represents one complete stimulus presentation session

3. **Signal Processing**:
   - Bandpass filtering (1-50 Hz) to remove noise and non-physiological frequencies
   - Notch filtering at 60 Hz to remove power line interference
   - Artifact removal by zeroing channels exceeding threshold (150 μV)

### 3.2 Feature Extraction

Two feature extraction methods are supported:

1. **Band Power Features** (enabled by default):
   - Extracts power in four frequency bands for each EEG channel:
     - Delta (1-4 Hz): Associated with deep sleep, unconscious processes
     - Theta (4-8 Hz): Associated with drowsiness, meditation, creativity
     - Alpha (8-13 Hz): Associated with relaxation, closed eyes
     - Beta (13-30 Hz): Associated with active thinking, focus
   - Results in a feature vector of length `NUM_CHANNELS × 4 = 64` features

2. **Raw Signal Flattening** (optional):
   - Directly flattens filtered signals, preserving temporal information
   - Results in a feature vector of length `NUM_CHANNELS × SUBTRIAL_SAMPLES = 16 × 250 = 4000` features

### 3.3 Data Augmentation

To improve model robustness, data augmentation is applied:

1. **Signal Augmentation (augment_trial)**:
   - Time-shifting: Randomly shifts signals by -30 to +30 samples
   - Amplitude scaling: Randomly scales signal intensity by 0.5× to 1.5×
   - Adding noise: Adds small Gaussian noise (σ = 10⁻⁴)
   - Signal inversion: Randomly inverts channel polarity (30% probability)
   - Channel swapping: Randomly swaps two channels (30% probability)
   - Creates N_AUG (default: 10) augmented versions of each trial

2. **Oversampling**:
   - Makes additional copies of samples to balance dataset (OVERSAMPLE_FACTOR = 2)

### 3.4 Model Training

The system uses a one-vs-all approach with binary classifiers:

1. **Feature Scaling**:
   - A StandardScaler is fit to the entire feature set
   - Transforms features to have zero mean and unit variance
   - The scaler is saved as `scaler.joblib` for use during trial/inference

2. **Binary Classification**:
   - For each class (0-11), a binary classifier is trained to distinguish:
     - Samples from that class (positive)
     - Samples from all other classes (negative)
   - Uses stratified k-fold cross-validation to assess model performance
   - Supports multiple classifier types:
     - Support Vector Machines (SVM)
     - Random Forests (RF)
     - Logistic Regression (LR)
     - Multi-layer Perceptron (MLP)
     - Custom classifiers (HybridRandomClassifier, HashedRandomClassifier)

3. **Classifier Selection**:
   - If TRY_ALL_MODELS is True, evaluates multiple classifier types and selects best
   - Otherwise uses SINGLE_MODEL (default: Random Forest with specific parameters)
   - Models are evaluated based on classification reports, with attention to bias

4. **Model Storage**:
   - Each binary classifier is saved to the "models" directory
   - Filename format: `eeg_image_classifier_class_{class_number}.joblib`
   - 12 separate binary classifier models are saved (one per class)

### 3.5 Custom Classifiers

Two custom classifier implementations are provided:

1. **HybridRandomClassifier**:
   - Combines a trained classifier with controlled randomization
   - Uses a parameter α (default: 0.5) to blend model predictions with random values
   - Useful for applications where deterministic but somewhat random outputs are needed

2. **HashedRandomClassifier**:
   - Produces reproducible, pseudo-random classifications based on input hashing
   - Doesn't actually learn from data but generates consistent responses for identical inputs
   - Useful for testing system behavior with predictable randomness

## 4. Real-time Classification (trial.py)

### 4.1 System Initialization

1. **Model Loading**:
   - Loads the 12 binary classifier models from the "models" directory
   - Loads the scaler from `scaler.joblib` for feature normalization

2. **Interface Setup**:
   - Creates a Tkinter GUI with:
     - Bar chart for visualization of class probabilities
     - Image display for the predicted class
     - EEG signal display (scrolling)

3. **Hardware Connection (optional)**:
   - If SIMULATOR_MODE is False, connects to OpenBCI board with same configuration as data collection
   - If SIMULATOR_MODE is True, loads simulation data from SIM_FILE

### 4.2 Real-time Processing Loop

The system continuously processes incoming EEG data through the RealTimeClassifierApp:

1. **Data Acquisition**:
   - In simulator mode: Generates 4-second chunks from pre-recorded data
   - In hardware mode: Pulls real-time data from OpenBCI board

2. **Signal Processing**:
   - Applies identical filters to training (bandpass, notch, artifact removal)
   - Generates feature vectors using same methods as training

3. **Prediction**:
   - Scales features using loaded scaler
   - For each binary classifier (class 0-11):
     - Gets prediction probability for the positive class
   - Combines all binary classifier outputs into a 12-class probability distribution

4. **Visualization**:
   - Updates bar chart showing probability for each class
   - Displays image corresponding to highest probability class
   - Refreshes at REFRESH_INTERVAL (default: 200ms)

### 4.3 Implementation Details

1. **RealTimeClassifierApp Class**:
   - Main application class handling interface and processing
   - `update_image_bars()`: Updates bar chart with probabilities
   - `update_eeg_plot()`: Updates scrolling EEG plot
   - `process_big_trial()`: Processes EEG chunk, extracts features, makes prediction
   - `check_for_subtrial()`: Scheduled function that checks for new data

2. **Simulation Mode**:
   - `load_sim_data()`: Loads simulation data from file
   - `generate_subtrial_sim()`: Extracts chunks from simulation data

## 5. Data Flow Across Components

### 5.1 From StimPresInt.py to train.py

1. **Data Format**:
   - StimPresInt.py generates text files with sample numbers and 16 EEG channels
   - Files are named according to subject ID, class number, and trial number
   - Example: `SM001_7_3.txt` for Subject SM001, Class 7, Trial 3

2. **Directory Structure**:
   - Files are organized in subject folders within the "All Nos" directory
   - This hierarchical organization allows training on data from multiple subjects

### 5.2 From train.py to trial.py

1. **Model Files**:
   - train.py generates 12 binary classifier models, one per class
   - These are saved in the "models" directory as .joblib files
   - The feature scaler is saved separately as `scaler.joblib`

2. **Configuration Consistency**:
   - Both train.py and trial.py use the same:
     - Frequency bands for feature extraction
     - Artifact threshold values
     - Signal processing parameters (bandpass range, notch frequency)
     - Feature extraction methods (band power, raw flattening)

### 5.3 Real-time Data Flow in trial.py

1. **Data Flow**:
   - Raw EEG data (16 channels) → Signal Processing → Feature Extraction → Feature Scaling → Binary Classification → Probability Combining → Visualization

2. **Timing**:
   - The system processes data in chunks similar to training (4-second windows)
   - Updates visualization at fixed intervals (REFRESH_INTERVAL = 200ms)

## 6. System Configuration and Parameters

### 6.1 Hardware Configuration

- **EEG Device**: OpenBCI Cyton+Daisy board (16 channels)
- **Connection Parameters**:
  - WiFi connection (default IP: 192.168.4.1, port: 6789)
  - Sample rate: 250 Hz (configured during setup)
  - Mode: Marker mode (for synchronization)

### 6.2 Data Collection Parameters

- **Visual Stimuli**: 6 distinct images, each with 2 variations (12 total classes)
- **Presentation Duration**: 5 seconds per stimulus
- **Recording Duration**: 2 seconds per subtrial, 10 subtrials per block
- **Naming Convention**: `[SubjectID]_[ClassNumber]_[TrialNumber].txt`

### 6.3 Signal Processing Parameters

- **Bandpass Filter**: 1-50 Hz, 4th order Butterworth
- **Notch Filter**: 60 Hz, Q=30 (removes power line interference)
- **Artifact Threshold**: 150 μV (channels exceeding this are zeroed)
- **Frequency Bands**:
  - Delta: 1-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-30 Hz

### 6.4 Training Parameters

- **Features**: Band power (default) and/or raw flattening (optional)
- **Augmentation**: 10 augmented samples per trial
- **Oversampling**: 2× oversampling factor
- **Default Model**: Random Forest with 200 trees, max depth 10
- **Model Selection Criterion**: Low bias (measured by class distribution imbalance)

### 6.5 Trial/Classification Parameters

- **Simulation Mode**: True (default) - uses pre-recorded data instead of live acquisition
- **Refresh Rate**: 200 ms (5 Hz update rate)
- **Image Display**: Shows images associated with predicted classes
- **Visualization**: Bar chart showing probability for each class

## 7. Concepts and Techniques

### 7.1 EEG Signal Processing

- **Bandpass Filtering**: Removes frequencies outside physiological range to reduce noise
- **Notch Filtering**: Specifically removes power line interference (60 Hz in US)
- **Artifact Removal**: Eliminates channels with extreme values from analysis
- **Band Power**: Measures signal power in specific frequency bands associated with brain states

### 7.2 Machine Learning Approach

- **One-vs-All Classification**: Trains binary classifier for each class, combines results
- **Feature Scaling**: Normalizes feature values to improve model performance
- **Data Augmentation**: Creates artificial variations of limited data to improve generalization
- **Cross-Validation**: Assesses model performance on different data splits
- **Model Selection**: Tests multiple classifier types to find optimal performance

### 7.3 Real-time Processing

- **Streaming Data Processing**: Continuously processes incoming data chunks
- **Feature Consistency**: Uses identical feature extraction between training and inference
- **Probability Visualization**: Visual feedback of classification confidence
- **Simulation Mode**: Allows system testing without hardware

## 8. System Limitations and Considerations

### 8.1 Technical Limitations

- **Signal Quality**: EEG is highly susceptible to noise and artifacts
- **Spatial Resolution**: Limited by number of electrodes (16 channels)
- **Feature Extraction**: Simple band power features may not capture all relevant patterns
- **Binary Classification**: One-vs-all approach may not be optimal for multi-class problems

### 8.2 Practical Considerations

- **Subject Variability**: EEG patterns vary significantly between individuals
- **Session Effects**: Data quality can change during a recording session (electrode drift)
- **Class Imbalance**: Number of samples per class may vary based on protocol execution
- **Temporal Dynamics**: Current implementation doesn't model temporal patterns across subtrials

### 8.3 Performance Factors

- **Signal-to-Noise Ratio**: Better SNR improves classification performance
- **Feature Selection**: Band power in common frequency bands may not be optimal for all tasks
- **Model Complexity**: Trade-off between model complexity and generalization
- **Real-time Constraints**: Processing must complete within refresh interval

## 9. Summary of Data Flow

1. **Data Collection**:
   - Subject views stimulus images through PowerPoint presentation
   - OpenBCI records synchronized EEG data with markers
   - Data saved as text files with subject/class/trial identifiers

2. **Data Processing**:
   - Text files loaded and parsed into trials
   - Signals filtered and artifacts removed
   - Features extracted (band power and/or raw signals)
   - Data augmented and oversampled

3. **Model Training**:
   - Features normalized using StandardScaler
   - Binary classifiers trained for each class
   - Models evaluated and selected based on performance
   - Final models and scaler saved to disk

4. **Real-time Classification**:
   - Models and scaler loaded
   - EEG data acquired (live or simulated)
   - Signals processed and features extracted
   - Binary classifiers predict class probabilities
   - Probabilities visualized in real-time interface

This complete pipeline represents an end-to-end brain-computer interface system for classifying visual stimulus response patterns in EEG data. 