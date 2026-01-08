# Heart Murmur Detection from Phonocardiogram Recordings

## Project Overview

This project implements a machine learning pipeline for detecting abnormal heart conditions from phonocardiogram (heart sound) recordings. The goal is to classify patient recordings as **Normal** or **Abnormal** based on audio analysis, which can serve as a screening tool for early detection of heart conditions.

### Purpose

The primary objective is to develop a CNN-based model that:
- Takes heart sound recordings (WAV files) as input
- Processes audio into time-frequency representations (spectrograms)
- Classifies patients as Normal or Abnormal
- Provides patient-level predictions through proper aggregation of multiple recordings

### Dataset

This project uses the **George B. Moody PhysioNet Challenge 2022** dataset:

**Citation:**
```
Reyna, M., Kiarashi, Y., Elola, A., Oliveira, J., Renna, F., Gu, A., 
Perez Alday, E. A., Sadr, N., Mattos, S., Coimbra, M., Sameni, R., 
Bahrami Rad, A., Koscova, Z., & Clifford, G. (2023). 
Heart Murmur Detection from Phonocardiogram Recordings: 
The George B. Moody PhysioNet Challenge 2022 (version 1.0.0). 
PhysioNet. https://doi.org/10.13026/t49p-5v35
```

**Dataset Details:**
- **942 patients** (486 Normal, 456 Abnormal)
- **3,163 WAV files** (multiple recordings per patient from different valve locations)
- **Sampling rate:** 4,000 Hz
- **Variable duration:** Recordings range from ~5 to 30+ seconds
- **Multiple auscultation locations:** Aortic Valve (AV), Pulmonary Valve (PV), Tricuspid Valve (TV), Mitral Valve (MV)

**Dataset Access:**
- Available at: https://physionet.org/content/challenge-2022/1.0.0/
- License: Creative Commons Attribution 4.0 International

### About the Author

**Ehsan Faghih**  
Ph.D. Student, North Carolina State University

This project is developed as a **self-learning exercise** in machine learning, specifically focusing on:
- Audio signal processing
- Deep learning for medical applications
- Time-series classification
- Multi-instance learning (patient-level aggregation)

While this is primarily for educational purposes and personal practice in the ML domain, the methods and pipeline developed here may be useful for future research or applications in medical signal processing.

---

## Project Structure

This project is organized into modular steps, each building upon the previous one to create a complete machine learning pipeline.

### Step-by-Step Pipeline Explanation

#### **Steps 1-4: Data Exploration and Setup**

**Step 1: Explore Data (`step1_explore_data.py`)**
- **Purpose:** Understand the dataset structure and basic statistics
- **Main Steps:**
  - Load patient metadata from CSV file
  - Count total patients and WAV files
  - Display outcome distribution (Normal vs Abnormal)
- **Key Functions:**
  - `pd.read_csv()` - Load patient metadata
  - `glob.glob()` - Find all WAV files
- **Output:** Dataset statistics showing 942 patients, 3,163 WAV files, balanced classes

**Step 2: Check Audio Properties (`step2_check_audio_properties.py`)**
- **Purpose:** Verify audio file characteristics (sampling rate, duration)
- **Main Steps:**
  - Read header files (.hea) to extract metadata
  - Extract sampling rate and number of samples
  - Calculate recording duration
- **Key Functions:**
  - File I/O to read .hea header files
  - Parse header format: `record_name num_signals sampling_freq num_samples`
- **Output:** Confirms 4,000 Hz sampling rate, variable durations

**Step 3: Load Labels (`step3_load_labels.py`)**
- **Purpose:** Extract patient outcome labels (Normal/Abnormal) from .txt files
- **Main Steps:**
  - Read patient .txt files
  - Parse lines starting with "#Outcome:"
  - Extract and return outcome label
- **Key Functions:**
  - `get_patient_outcome(patient_id, data_dir)` - Returns 'Normal' or 'Abnormal'
- **Output:** Function to map patient IDs to their outcome labels

**Step 4: Create Dataset Mapping (`step4_create_dataset_mapping.py`)**
- **Purpose:** Build complete mapping of WAV files to patient outcomes
- **Main Steps:**
  - Find all WAV files in dataset directory
  - Extract patient ID from filename (format: `{PatientID}_{Location}.wav`)
  - Get outcome label for each patient
  - Create list of (wav_file_path, patient_id, outcome) tuples
- **Key Functions:**
  - `create_dataset_mapping(data_dir)` - Returns list of all WAV files with labels
- **Output:** Complete dataset structure: 3,163 WAV files mapped to 942 patients

---

#### **Steps 5-7: Audio Preprocessing Pipeline**

**Step 5: Preprocess Audio (`step5_preprocess_audio.py`)**
- **Purpose:** Load and normalize audio files for consistent processing
- **Main Steps:**
  - Load WAV file using scipy.io.wavfile
  - Convert to float32 format
  - Normalize amplitude to [-1, 1] range
- **Key Functions:**
  - `preprocess_audio(wav_file, target_sr=4000)` - Returns normalized audio array and sampling rate
- **Output:** Preprocessed audio ready for feature extraction

**Step 6: Create Spectrograms (`step6_create_spectrograms.py`)**
- **Purpose:** Convert audio signals to time-frequency representations (spectrograms)
- **Main Steps:**
  - Compute power spectrogram using scipy.signal.spectrogram
  - Apply log transformation (log10) to compress dynamic range
  - Create log-mel spectrogram representation
- **Key Functions:**
  - `create_log_mel_spectrogram(audio, sr, n_mels=64, hop_length=512)` - Returns spectrogram array
- **Output:** 2D spectrogram arrays (frequency bins × time frames) suitable for CNN input

**Step 7: Split into Clips (`step7_split_into_clips.py`)**
- **Purpose:** Split variable-length recordings into fixed-length clips for uniform model input
- **Main Steps:**
  - Calculate clip length in samples (4 seconds × sampling rate)
  - Create overlapping windows (50% overlap)
  - Extract clips with hop size
  - Handle remaining audio if significant (>2 seconds)
- **Key Functions:**
  - `split_into_clips(audio, sr, clip_duration=4, overlap=0.5)` - Returns list of clip arrays
- **Output:** Uniform 4-second clips with 50% overlap (e.g., 23.6s recording → 11 clips)

---

#### **Steps 8-11: Dataset Preparation and Planning**

**Step 8: Create Full Dataset Pipeline (`step8_create_full_dataset.py`)**
- **Purpose:** Combine all preprocessing steps into complete pipeline
- **Main Steps:**
  - Process WAV files through: load → normalize → split → spectrogram
  - Test pipeline on sample files
  - Verify end-to-end functionality
- **Key Functions:**
  - `process_wav_to_clips(wav_file, clip_duration=4, overlap=0.5)` - Complete pipeline function
- **Output:** Verified pipeline that converts WAV → clips → spectrograms

**Step 9: Setup Cross-Validation (`step9_setup_cross_validation.py`)**
- **Purpose:** Configure patient-level cross-validation to prevent data leakage
- **Main Steps:**
  - Extract unique patients and their outcomes
  - Setup 5-fold stratified cross-validation
  - Ensure patient-level splits (not clip-level)
- **Key Functions:**
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` - Creates CV splits
- **Output:** Patient-level train/test splits ensuring no patient appears in both sets

**Step 10: Build CNN Model (`step10_build_cnn_model.py`)**
- **Purpose:** Define CNN architecture structure (conceptual)
- **Main Steps:**
  - Define model layers (Conv2D, MaxPooling, Dense)
  - Specify input/output shapes
  - Document architecture choices
- **Key Functions:**
  - `create_simple_cnn(input_shape)` - Returns model layer description
- **Output:** Model architecture specification ready for implementation

**Step 11: Training Plan (`step11_training_plan.py`)**
- **Purpose:** Outline complete training and evaluation workflow
- **Main Steps:**
  - Document training steps
  - Define evaluation strategy
  - Specify key principles (patient-level evaluation, aggregation)
- **Output:** Complete workflow roadmap

---

#### **Steps 12-15: Model Implementation and Evaluation**

**Step 12: Implement CNN Model (`step12_implement_cnn_model.py`)**
- **Purpose:** Create actual trainable CNN model using TensorFlow/Keras
- **Main Steps:**
  - Build Sequential model with Conv2D layers
  - Add pooling and dense layers
  - Compile with optimizer, loss, and metrics
- **Key Functions:**
  - `create_cnn_model(input_shape=(1025, 181, 1))` - Returns compiled Keras model
  - Architecture: 2 Conv2D blocks → Flatten → Dense(128) → Dropout → Dense(1, sigmoid)
- **Output:** Ready-to-train CNN model

**Step 13: Prepare Training Data (`step13_prepare_training_data.py`)**
- **Purpose:** Process all WAV files into training-ready format
- **Main Steps:**
  - Load all WAV files from dataset
  - Process each file: audio → clips → spectrograms
  - Assign binary labels (0=Normal, 1=Abnormal)
  - Track patient IDs for each clip
- **Key Functions:**
  - `prepare_all_clips(dataset, max_files=None)` - Returns (X, y, patient_ids)
- **Output:** NumPy arrays of spectrograms, labels, and patient IDs

**Step 14: Patient Aggregation (`step14_patient_aggregation.py`)**
- **Purpose:** Aggregate clip-level predictions to patient-level predictions
- **Main Steps:**
  - Group clip predictions by patient ID
  - Aggregate using mean or max pooling
  - Return patient-level probabilities
- **Key Functions:**
  - `aggregate_clips_to_patient(clip_predictions, patient_ids, method='mean')` - Returns dict of patient predictions
  - Methods: 'mean' (average) or 'max' (maximum) pooling
- **Output:** Patient-level predictions from clip-level predictions

**Step 15: Evaluation Metrics (`step15_evaluation_metrics.py`)**
- **Purpose:** Calculate comprehensive patient-level classification metrics
- **Main Steps:**
  - Convert probabilities to binary predictions (threshold=0.5)
  - Calculate confusion matrix
  - Compute ROC-AUC, F1, Sensitivity, Specificity, Accuracy
- **Key Functions:**
  - `calculate_patient_metrics(y_true, y_pred_proba, threshold=0.5)` - Returns metrics dictionary
  - Uses sklearn.metrics for calculations
- **Output:** Complete evaluation metrics for model performance

---

#### **Steps 16-17: Complete Pipeline and Summary**

**Step 16: Full Training Loop (`step16_full_training_loop.py`)**
- **Purpose:** Combine all components into complete training pipeline structure
- **Main Steps:**
  - Setup cross-validation loop
  - For each fold: prepare data → train → predict → aggregate → evaluate
  - Collect results across folds
- **Key Functions:**
  - `train_and_evaluate_fold(train_patients, test_patients, dataset_dict)` - Complete fold processing
- **Output:** Training loop structure (placeholder for actual TensorFlow implementation)

**Step 17: Summary and Next Steps (`step17_summary_and_next_steps.py`)**
- **Purpose:** Document completed work and provide roadmap
- **Main Steps:**
  - List all completed components
  - Document what's been built
  - Outline next steps for actual training
- **Output:** Project summary and implementation guide

---

## Other Files

### Training Scripts

**`train_complete.py`** - **Main Training Script (RECOMMENDED)**
- Complete implementation of training pipeline
- Processes full dataset with configurable options
- Implements 5-fold cross-validation
- Compares mean vs max aggregation methods
- Reports comprehensive metrics
- **Configuration options:**
  - `USE_FULL_DATASET`: True/False for full or limited dataset
  - `MAX_PATIENTS_PER_FOLD`: Limit number of patients per fold
  - `HYPERPARAMS`: Epochs, batch size, learning rate

**`train_model.py`** - Original Training Script
- Simpler version using sample data
- Good for quick testing
- Processes subset of patients for faster execution

**`train_model_full.py`** - Alternative Full Implementation
- Alternative complete implementation
- Similar to train_complete.py with different structure

### Configuration and Documentation

**`TRAINING_GUIDE.md`** - **Complete Training Instructions**
- Step-by-step guide for running training
- Configuration options explained
- Troubleshooting tips
- Time estimates
- Hyperparameter tuning guide
- **READ THIS FILE for detailed running instructions**

**`requirements.txt`** - Python Dependencies
- Lists required packages (numpy, scipy, pandas, scikit-learn, tensorflow)
- Can be used with `pip install -r requirements.txt`

**`install_dependencies.py`** - Dependency Installation Helper
- Script to install packages one by one
- Helps identify installation issues
- Useful for troubleshooting

### Utility Scripts

**`check_installed_packages.py`** - Package Verification
- Checks which required packages are installed
- Verifies imports work correctly
- Useful for environment setup

**`test_training.py`** - Quick Test Script
- Minimal test to verify pipeline works
- Uses very small dataset
- Quick validation before full training

---

## Project Workflow Summary

```
1. Data Exploration (Steps 1-4)
   └─> Understand dataset → Map files to labels

2. Preprocessing (Steps 5-7)
   └─> Load audio → Create spectrograms → Split into clips

3. Model Setup (Steps 8-12)
   └─> Prepare data → Setup CV → Build CNN model

4. Training & Evaluation (Steps 13-17)
   └─> Train model → Aggregate predictions → Evaluate metrics
```

---

## Key Design Decisions

1. **Patient-Level Evaluation:** Model trains on clips but evaluates on patients (correct evaluation level)

2. **Patient-Level Splits:** Cross-validation splits by patients, not clips (prevents data leakage)

3. **Aggregation Methods:** Compares mean (stable) vs max (sensitive) pooling for patient predictions

4. **Spectrogram Representation:** Uses log-mel spectrograms (standard for audio classification)

5. **Fixed-Length Clips:** Splits variable-length recordings into 4-second clips with overlap

6. **Class Balance:** Dataset is relatively balanced (486 Normal, 456 Abnormal)

---

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Required packages: numpy, scipy, pandas, scikit-learn, tensorflow

### Quick Start

1. **Setup Environment:**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   pip install numpy scipy pandas scikit-learn tensorflow
   ```

2. **Download Dataset:**
   - Download from PhysioNet: https://physionet.org/content/challenge-2022/1.0.0/
   - Extract to `dataset/training_data/` directory
   - Ensure `dataset/training_data.csv` is present

3. **Run Steps 1-17:**
   - Execute each step file sequentially: `python step1_explore_data.py`
   - Each step builds on previous ones
   - Steps are designed to be run independently

4. **Run Training:**
   - **For detailed instructions, see `TRAINING_GUIDE.md`**
   - Quick test: Edit `train_complete.py` → Set `USE_FULL_DATASET = False` → Run
   - Full training: Set `USE_FULL_DATASET = True` → Run (takes hours)

---

## Results and Metrics

The model is evaluated using patient-level metrics:
- **ROC-AUC:** Area under ROC curve (overall performance)
- **F1 Score:** Harmonic mean of precision and recall
- **Sensitivity:** True positive rate (detecting Abnormal cases)
- **Specificity:** True negative rate (detecting Normal cases)
- **Accuracy:** Overall classification accuracy

Both **mean** and **max** aggregation methods are compared to determine the best approach for patient-level predictions.

---

## Future Improvements

Potential enhancements for future work:
1. **Multi-instance Learning:** Attention-based aggregation across clips
2. **Better Spectrograms:** Proper mel-scale spectrograms using librosa
3. **Data Augmentation:** Time stretching, noise addition, pitch shifting
4. **Advanced Architectures:** ResNet, EfficientNet, or Transformer-based models
5. **Hyperparameter Optimization:** Automated tuning (GridSearch, Bayesian optimization)
6. **Ensemble Methods:** Combine multiple models for better performance
7. **Calibration:** Probability calibration for better uncertainty estimation

---

## Citation

If you use this code or are inspired by this project, please cite:

1. **Dataset:**
   ```
   Reyna, M., et al. (2023). Heart Murmur Detection from Phonocardiogram 
   Recordings: The George B. Moody PhysioNet Challenge 2022 (version 1.0.0). 
   PhysioNet. https://doi.org/10.13026/t49p-5v35
   ```

2. **PhysioNet Platform:**
   ```
   Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: 
   Components of a new research resource for complex physiologic signals. 
   Circulation, 101(23), e215-e220.
   ```

---

## License

This project is for educational and research purposes. The dataset is licensed under Creative Commons Attribution 4.0 International. Please refer to the original dataset license for usage terms.

---

## Contact

**Ehsan Faghih**  
Ph.D. Student  
North Carolina State University

*This project is developed as a self-learning exercise in machine learning and medical signal processing.*

---

## How to Run This Project

**For detailed step-by-step instructions on running the training pipeline, please refer to `TRAINING_GUIDE.md`**

The training guide includes:
- Quick test instructions
- Full training setup
- Configuration options
- Hyperparameter tuning guide
- Troubleshooting tips
- Time estimates

---

## Acknowledgments

- PhysioNet Challenge 2022 organizers for providing the dataset
- NC State University for academic support
- Open-source ML community for tools and libraries

