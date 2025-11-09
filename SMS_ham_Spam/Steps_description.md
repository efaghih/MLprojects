# SMS Spam Classification - Project Development Steps

This document provides a detailed, step-by-step account of how this SMS spam classification project was built, from initial setup to final deliverables.

---

## ⚠️ Important Notice About Model Files

**The trained models in the `models/` directory are NOT included in this repository** because they exceed 1GB in total size:
- Baseline model: ~500KB
- DistilBERT checkpoint: ~268MB (primary contributor to size)
- Additional checkpoint files and logs

**All models are fully reproducible** by running the training scripts documented in this file. The code, data processing pipeline, and training configurations are provided to ensure you can regenerate identical models.

To reproduce the models:
1. Run `python train_baseline.py` (takes ~2 seconds)
2. Run `python train_distilbert.py` (takes ~2 hours on CPU)

---

## Table of Contents
1. [Repository and Environment Setup](#1-repository-and-environment-setup)
2. [Data Download and Initial Exploration](#2-data-download-and-initial-exploration)
3. [Data Cleaning and Preprocessing](#3-data-cleaning-and-preprocessing)
4. [Stratified Train/Val/Test Split](#4-stratified-trainvaltest-split)
5. [Baseline Model Training](#5-baseline-model-training)
6. [Modern Baseline: DistilBERT](#6-modern-baseline-distilbert)
7. [Model Evaluation on Test Set](#7-model-evaluation-on-test-set)
8. [Model Comparison and Analysis](#8-model-comparison-and-analysis)
9. [Interactive Testing Interface](#9-interactive-testing-interface)
10. [Documentation and Model Card](#10-documentation-and-model-card)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Repository and Environment Setup

### Purpose
Establish a clean, reproducible development environment with all necessary dependencies.

### Key Decisions
- Use virtual environment for dependency isolation
- Pin package versions for reproducibility
- Support both classical ML (scikit-learn) and deep learning (PyTorch/Transformers)

### Commands Used
```bash
# Create project directory
mkdir SMS_ham_Spam
cd SMS_ham_Spam

# Create virtual environment (optional but recommended)
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Outputs
- **File Created**: `requirements.txt`
  - Core ML: pandas, numpy, scikit-learn, joblib
  - Visualization: matplotlib, seaborn
  - Deep Learning: torch, transformers, datasets, evaluate, accelerate
  - Development: jupyter, ipykernel

### Location
- Root directory: `SMS_ham_Spam/`
- Dependencies: `requirements.txt`

---

## 2. Data Download and Initial Exploration

### Purpose
Obtain the SMS Spam Collection dataset and understand its structure.

### Key Decisions
- Use the UCI SMS Spam Collection dataset (5,572 messages)
- Keep raw data separate from processed data
- Maintain original file for reference

### Data Source
- Dataset: SMS Spam Collection from UCI Machine Learning Repository
- Format: Tab-separated file with two columns (label, text)
- Raw file: `Dataset/SMSSpamCollection`

### Initial Statistics
- Total messages: 5,572
- Ham (legitimate): 4,825 (86.59%)
- Spam: 747 (13.41%)
- Class imbalance ratio: ~6.5:1

### Outputs
- **Raw data stored in**: `Dataset/SMSSpamCollection`
- **Dataset readme**: `Dataset/readme`

---

## 3. Data Cleaning and Preprocessing

### Purpose
Convert raw text data into a clean, ML-ready format with proper labels.

### Key Decisions
- Convert string labels ("ham"/"spam") to binary integers (0/1)
- Keep text in original form (no aggressive preprocessing at this stage)
- Preserve all messages (no filtering)
- Use UTF-8 encoding to handle special characters

### Implementation
**Module**: `src/data.py`

Key functions:
- `load_raw_data()`: Load tab-separated SMS data
- `create_binary_labels()`: Convert labels to 0 (ham) and 1 (spam)

### Script
```bash
# Run data processing pipeline
python make_data.py
```

### Outputs
- **Processed files**: `data/processed/`
  - Clean CSV format with columns: `label`, `text`
  - Binary labels (0=ham, 1=spam)

---

## 4. Stratified Train/Val/Test Split

### Purpose
Create reproducible data splits that maintain class distribution across all sets.

### Key Decisions
- **Split ratio**: 80/10/10 (train/validation/test)
- **Stratification**: Maintain 13.41% spam ratio in all splits
- **Random seed**: Fixed at 42 for reproducibility
- **Validation set purpose**: Model selection and hyperparameter tuning
- **Test set**: Frozen for final evaluation only

### Why Stratified?
With only 13.41% spam, random splitting could create unbalanced sets. Stratification ensures each split has the same spam/ham ratio.

### Implementation
**Function**: `stratified_split()` in `src/data.py`

Uses scikit-learn's `train_test_split` with `stratify` parameter in two stages:
1. Split train from (val + test)
2. Split val from test

### Commands
```bash
python make_data.py
```

### Outputs
**Location**: `data/processed/`

| File | Samples | Ham | Spam | Spam % |
|------|---------|-----|------|--------|
| `train.csv` | 4,457 | 3,859 | 598 | 13.42% |
| `val.csv` | 557 | 483 | 74 | 13.29% |
| `test.csv` | 558 | 483 | 75 | 13.44% |

**Verification**: All splits maintain ~13.4% spam ratio ✓

---

## 5. Baseline Model Training

### Purpose
Establish a fast, interpretable baseline using classical machine learning techniques.

### Key Decisions

#### Model Architecture
**Pipeline**: TF-IDF Vectorizer → Multinomial Naive Bayes

#### TF-IDF Configuration
- `ngram_range=(1, 2)`: Capture both single words and two-word phrases
  - Example: "free", "click here", "call now"
- `min_df=2`: Ignore terms appearing in fewer than 2 documents (reduces noise from typos)
- `stop_words='english'`: Remove common words (the, is, are, etc.)
- `lowercase=True`: Normalize text casing

#### Naive Bayes Configuration
- `alpha=0.5`: Laplace smoothing to handle unseen words
- Model choice: Multinomial NB (optimal for text with word counts)

#### Training Strategy
- Train on training set (4,457 samples)
- Validate on validation set (557 samples)
- Report metrics for both to detect overfitting

### Why This Baseline?
1. **Fast**: Trains in ~2 seconds
2. **Interpretable**: Can examine feature weights
3. **Proven**: Industry standard for text classification
4. **Lightweight**: Small model size (~500KB)

### Implementation
**Module**: `src/train_baseline.py`

Key functions:
- `create_baseline_pipeline()`: Build TF-IDF + NB pipeline
- `train_model()`: Fit on training data
- `evaluate_model()`: Compute comprehensive metrics
- `save_model()`: Persist with joblib

### Commands
```bash
python train_baseline.py
```

### Metrics Computed
- **F1 Score**: Harmonic mean of precision/recall (spam as positive class)
- **ROC-AUC**: Area under ROC curve (discrimination ability)
- **PR-AUC**: Precision-Recall AUC (better for imbalanced data)
- **Confusion Matrix**: Breakdown of true/false positives/negatives
- **Classification Report**: Per-class precision, recall, F1

### Outputs
**Location**: `models/baseline/`

> **Note:** These files are generated when you run the training script and are **not included in the repository** due to size constraints.

1. **`baseline_model.pkl`** (500KB)
   - Complete scikit-learn pipeline
   - Includes fitted TF-IDF vectorizer (7,370 terms) and NB classifier

2. **`metrics.txt`**
   - Training and validation metrics
   - Timestamp for version tracking

3. **`confusion_matrices.png`**
   - Side-by-side visualization of train vs validation performance

### Validation Set Results
```
Accuracy:  97.49%
F1 Score:  0.8955
Precision: 1.0000 (no false positives!)
Recall:    0.8108
ROC-AUC:   0.9842
PR-AUC:    0.9576
```

### Key Finding
Slight overfitting detected (F1 gap: 6.63%), but acceptable for baseline.

---

## 6. Modern Baseline: DistilBERT

### Purpose
Implement a state-of-the-art transformer model for comparison with classical ML.

### Key Decisions

#### Model Selection: DistilBERT
- **Base model**: `distilbert-base-uncased`
- **Why DistilBERT?**
  - 40% smaller than BERT
  - 60% faster inference
  - Retains 97% of BERT's performance
  - More practical for deployment

#### Hyperparameters (As Specified)
- `max_length=128`: Truncate sequences at 128 tokens
- `batch_size=16`: Balance between speed and memory
- `epochs=3`: Sufficient for fine-tuning
- `learning_rate=2e-5`: Small LR to avoid catastrophic forgetting
- `weight_decay=0.01`: L2 regularization
- `warmup_steps=100`: Linear warmup for stable training

#### Training Strategy
- **Transfer learning**: Start from pre-trained weights
- **Fine-tuning**: Update all parameters (no freezing)
- **Early stopping**: Patience=2 epochs on validation F1
- **Best model selection**: Load checkpoint with highest validation F1

### Why These Settings?
- **Max length 128**: SMS messages are short (avg ~80 characters)
- **Small learning rate**: Preserves pre-trained knowledge
- **Few epochs**: Prevents overfitting on small dataset

### Implementation
**Module**: `src/train_distilbert.py`

Key components:
- `prepare_tokenizer()`: Load DistilBERT tokenizer
- `tokenize_dataset()`: Convert text to input IDs and attention masks
- `load_model()`: Load pre-trained model for sequence classification
- `create_training_args()`: Configure Hugging Face Trainer
- `train_model()`: Fine-tune with early stopping
- `evaluate_model()`: Comprehensive evaluation on all splits

### Commands
```bash
# Install deep learning dependencies (if not already installed)
pip install torch transformers datasets evaluate accelerate

# Train model (takes ~2 hours on CPU, ~20 minutes on GPU)
python train_distilbert.py
```

### Training Process
```
Epoch 1: val_f1=0.9726, val_loss=0.0418
Epoch 2: val_f1=0.9726, val_loss=0.0403
Epoch 3: val_f1=0.9726, val_loss=0.0433
Best checkpoint: Epoch 2 (checkpoint-837)
```

### Outputs
**Location**: `models/distilbert/`

> **Note:** These files are generated when you run the training script and are **not included in the repository** due to their large size (~268MB for the checkpoint alone).

1. **`checkpoint-837/`** (268MB)
   - Best model checkpoint
   - Contains: `model.safetensors`, `config.json`, `optimizer.pt`

2. **`metrics.txt`**
   - Train/val/test metrics
   - Model configuration details

3. **`confusion_matrices.png`**
   - Performance across all three splits

4. **`logs/`**
   - TensorBoard-compatible training logs

### Validation Set Results
```
Accuracy:  99.28%
F1 Score:  0.9726
Precision: 0.9861
Recall:    0.9595
ROC-AUC:   0.9960
PR-AUC:    0.9854
```

### Key Findings
- Minimal overfitting (F1 gap: 0.97%)
- Excellent generalization
- Significant improvement over baseline on recall

---

## 7. Model Evaluation on Test Set

### Purpose
Perform final, unbiased evaluation on the held-out test set that was never used during training or hyperparameter selection.

### Key Principles
- **Test set is sacred**: Only evaluated once at the end
- **Same test set for both models**: Ensures fair comparison
- **No further tuning**: Accept results as-is

### Implementation
Both models evaluated independently on `data/processed/test.csv`

### Test Set Results

#### Baseline (TF-IDF + Naive Bayes)
```
Accuracy:        98.03%
F1 Score:        0.9209
Precision:       1.0000  ← Perfect! No false alarms
Recall:          0.8533
ROC-AUC:         0.9944
PR-AUC:          0.9810

Confusion Matrix:
              Predicted
              Ham   Spam
Actual  Ham   483     0  ← Zero false positives
        Spam   11    64
```

#### DistilBERT (Transformer)
```
Accuracy:        99.10%
F1 Score:        0.9664
Precision:       0.9730
Recall:          0.9600  ← Catches more spam
ROC-AUC:         0.9935
PR-AUC:          0.9880

Confusion Matrix:
              Predicted
              Ham   Spam
Actual  Ham   481     2  ← 2 false positives
        Spam    3    72  ← Only 3 missed spam
```

### Performance Gap Analysis
- DistilBERT catches **8 more spam messages** (11 → 3 false negatives)
- Trade-off: DistilBERT has **2 false positives** (baseline had 0)
- F1 improvement: **+4.56 percentage points**
- Recall improvement: **+12.5%** (relative)

---

## 8. Model Comparison and Analysis

### Purpose
Create a comprehensive, side-by-side comparison with visualizations to understand the strengths and weaknesses of each approach.

### Implementation
**Script**: `compare_models.py`

Features:
1. Load both trained models
2. Evaluate on same test set
3. Compute identical metrics
4. Generate comparison report
5. Create visualizations

### Commands
```bash
python compare_models.py
```

### Outputs
**Location**: `reports/`

#### 1. `model_comparison.txt`
Detailed text report with:
- Side-by-side metrics table
- Improvement calculations
- Confusion matrix breakdowns
- Interpretation notes

#### 2. `comparison_metrics.csv`
Machine-readable metrics table:
```csv
Metric,TF-IDF + NB,DistilBERT,Improvement,Improvement %
Accuracy,0.9803,0.9910,0.0108,1.0969
Precision,1.0000,0.9730,-0.0270,-2.7027
Recall,0.8533,0.9600,0.1067,12.5000
F1 Score,0.9209,0.9664,0.0456,4.9497
ROC-AUC,0.9944,0.9935,-0.0009,-0.0902
PR-AUC,0.9810,0.9880,0.0070,0.7160
```

#### 3. `model_comparison.png`
Four-panel visualization:
- **Panel 1**: Bar chart comparing all metrics
- **Panel 2**: Improvement chart (green=better, red=worse)
- **Panel 3**: Baseline confusion matrix
- **Panel 4**: DistilBERT confusion matrix

### Key Insights

#### Where Baseline Excels
✅ **Perfect precision** (1.0000) - No legitimate messages blocked
✅ **Fast inference** (~1ms per message)
✅ **Interpretable** - Can inspect feature weights
✅ **Lightweight** - 500KB model size

#### Where DistilBERT Excels
✅ **Higher recall** (0.96 vs 0.85) - Catches more spam
✅ **Better F1 score** (0.9664 vs 0.9209)
✅ **More balanced** - Better precision/recall trade-off
✅ **State-of-the-art** - Leverages pre-trained language understanding

#### Trade-off Analysis
**Use baseline if:**
- Zero false positives is critical (e.g., important messages)
- Inference speed matters
- Limited compute resources
- Model interpretability required

**Use DistilBERT if:**
- Maximum spam detection is priority
- Can tolerate rare false positives
- Have GPU for inference
- Want best overall performance

---

## 9. Interactive Testing Interface

### Purpose
Provide an easy way to test models on custom messages and verify predictions interactively.

### Implementation
**Script**: `test_model.py`

Features:
1. Load trained baseline model
2. Pre-loaded example messages (8 samples)
3. Interactive mode for custom input
4. Confidence scores displayed
5. User-friendly output with emojis

### Commands
```bash
python test_model.py
```

### Sample Output
```
Message 1:
  Text: "Hi! How are you doing today?"
  Prediction: ✅ HAM
  Confidence: 99.56%

Message 2:
  Text: "Congratulations! You've won a $1000 gift card. Call now!"
  Prediction: 🚫 SPAM
  Confidence: 80.06%

Message 3:
  Text: "Can we meet for lunch tomorrow at 12pm?"
  Prediction: ✅ HAM
  Confidence: 99.01%

🔮 Try your own messages! (Type 'quit' to exit)
Enter SMS message: FREE WIN NOW!!!
  → 🚫 SPAM (Confidence: 95.23%)
```

### Use Cases
- Quick sanity checks
- Demo to stakeholders
- Testing edge cases
- Model behavior exploration

---

## 10. Documentation and Model Card

### Purpose
Create comprehensive documentation for reproducibility, maintenance, and portfolio presentation.

### Documents Created

#### 1. `README.md`
**Sections:**
- Project overview and structure
- Dataset statistics
- Step-by-step usage instructions
- Model performance comparison
- Installation guide
- Model selection guide
- Best practices demonstrated

**Key Features:**
- Clear instructions for each step
- Performance tables
- Visual hierarchy with emojis
- Technical achievements highlighted
- Trade-offs explained

#### 2. `Steps_description.md` (This Document)
**Purpose:** Detailed technical documentation

**Content:**
- Chronological step-by-step process
- Purpose and rationale for each decision
- Commands and scripts used
- Output locations
- Key findings and insights

#### 3. Inline Code Documentation
**Standards:**
- Every function has docstring with:
  - Purpose description
  - Parameters with types
  - Return values
  - Usage examples where helpful
- Complex logic explained with comments
- Type hints for better IDE support

### Model Card Components

#### Model Details
- **Model Name**: SMS Spam Classifier
- **Model Type**: Text Classification (Binary)
- **Architecture Options**:
  - Classical: TF-IDF + Multinomial Naive Bayes
  - Modern: DistilBERT (distilbert-base-uncased)
- **Training Date**: November 2025
- **Version**: 1.0

#### Intended Use
- **Primary Use**: SMS spam detection
- **Target Users**: Email/messaging platforms, security teams
- **Out of Scope**: Not intended for email spam, social media, or other text types

#### Training Data
- **Dataset**: SMS Spam Collection (UCI ML Repository)
- **Size**: 5,572 messages
- **Split**: 80/10/10 (train/val/test)
- **Class Distribution**: 86.59% ham, 13.41% spam
- **Language**: English
- **Time Period**: 2012-2013 (dataset collection period)

#### Performance Metrics
See [Section 7](#7-model-evaluation-on-test-set) for detailed results.

#### Limitations
- **English only**: Model trained on English messages
- **Dataset age**: May not capture modern spam tactics
- **SMS specific**: Optimized for short messages (<160 chars typical)
- **Imbalanced data**: More ham examples than spam

#### Ethical Considerations
- **False positives**: May block legitimate messages (especially DistilBERT)
- **False negatives**: May allow spam through (especially baseline)
- **Bias**: May favor message types common in training data
- **Privacy**: Model processes message content

#### Recommendations
- **Threshold tuning**: Adjust decision threshold based on use case
- **Human review**: Implement feedback loop for edge cases
- **Monitoring**: Track performance drift over time
- **Updates**: Retrain periodically with new spam examples

---

## 11. Future Enhancements

The following steps would further strengthen this project but are not yet implemented. They represent natural next steps for production deployment or advanced analysis.

### 11.1 GridSearchCV Hyperparameter Tuning

#### Purpose
Systematically search for optimal hyperparameters without data leakage.

#### Proposed Approach
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__min_df': [2, 5, 10],
    'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5-fold CV on training set only
    scoring='f1',
    n_jobs=-1
)

# Fit on training set only (no test set leakage)
grid_search.fit(X_train, y_train)
```

#### Output Location
`models/baseline/grid_search_results.csv`

#### Key Principle
**Never use test set in GridSearchCV** - only training and validation sets.

---

### 11.2 Probability Calibration

#### Purpose
Ensure predicted probabilities reflect true likelihood of spam.

#### Approach
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_estimator=trained_pipeline,
    method='isotonic',  # or 'sigmoid'
    cv='prefit'  # Model already trained
)

# Calibrate on validation set
calibrated_model.fit(X_val, y_val)
```

#### Evaluation
- Plot calibration curves
- Compare before/after reliability diagrams
- Measure Brier score

#### Output Location
`models/baseline/calibrated_model.pkl`

---

### 11.3 Threshold Selection

#### Purpose
Optimize decision threshold for specific business requirements.

#### Approach
```python
from sklearn.metrics import precision_recall_curve

# Get probabilities on validation set
probas = model.predict_proba(X_val)[:, 1]

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_val, probas)

# Find threshold for desired operating point
# Example: 99% precision
threshold_99p = thresholds[np.argmax(precision >= 0.99)]
```

#### Use Cases
- **High precision threshold**: Minimize false positives (0.99 precision)
- **High recall threshold**: Maximize spam detection (0.95 recall)
- **Balanced threshold**: Optimize F1 score

#### Output Location
`reports/threshold_analysis.png`

---

### 11.4 Error Analysis

#### Purpose
Understand where and why models make mistakes.

#### Components

##### Confusion Matrix Deep Dive
- Analyze false positives: What legitimate messages look like spam?
- Analyze false negatives: What spam looks like legitimate?

##### Difficult Examples Analysis
```python
# Find low-confidence predictions
uncertain = np.abs(probas - 0.5) < 0.1

# Find misclassified examples
errors = y_pred != y_true

# Combine for detailed analysis
difficult_cases = X[uncertain | errors]
```

##### Error Patterns
- Length analysis (are short/long messages harder?)
- Keyword analysis (which spam keywords are missed?)
- Linguistic patterns (punctuation, capitalization, etc.)

#### Output Location
- `reports/error_analysis.txt`
- `reports/false_positives.csv`
- `reports/false_negatives.csv`

---

### 11.5 Robustness Checks

#### Purpose
Verify model stability under various conditions.

#### Tests

##### Adversarial Examples
```python
# Test intentional typos
"F R E E  m o n e y"  # Spacing
"Fr33 m0ney"  # Leetspeak
"Free mοney"  # Unicode tricks (o → ο)
```

##### Out-of-Distribution
- Modern slang (2024+)
- Emojis and special characters
- Multiple languages
- Very short/long messages

##### Perturbation Robustness
- Random word substitution
- Word reordering
- Synonym replacement

#### Output Location
`reports/robustness_tests.txt`

---

### 11.6 Explainability Artifacts

#### Purpose
Make model decisions interpretable and trustworthy.

#### For Classical Model (TF-IDF + NB)

##### Feature Importance
```python
# Get feature weights from trained model
feature_names = vectorizer.get_feature_names_out()
log_probs = nb_classifier.feature_log_prob_

# Top spam indicators
spam_weights = log_probs[1]  # Class 1
top_spam_words = sorted(zip(feature_names, spam_weights), 
                        key=lambda x: x[1], reverse=True)[:20]
```

##### LIME Explanations
```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['ham', 'spam'])

# Explain a prediction
explanation = explainer.explain_instance(
    text_instance,
    model.predict_proba,
    num_features=10
)
```

#### For DistilBERT

##### Attention Visualization
```python
# Extract attention weights
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions

# Visualize which tokens the model focuses on
plot_attention_heatmap(text, attentions)
```

##### Integrated Gradients
```python
from captum.attr import LayerIntegratedGradients

# Compute token-level attributions
lig = LayerIntegratedGradients(model, model.distilbert.embeddings)
attributions = lig.attribute(inputs, target=predicted_class)
```

#### Output Location
- `reports/explainability/`
  - `top_spam_features.txt`
  - `top_ham_features.txt`
  - `lime_examples.html`
  - `attention_visualizations/`

---

### 11.7 Packaging and Versioning

#### Purpose
Prepare models for production deployment with proper version control.

#### Components

##### Model Versioning
```python
import joblib
from datetime import datetime

model_metadata = {
    'version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_f1': test_f1_score,
    'dependencies': {
        'scikit-learn': sklearn.__version__,
        'python': sys.version
    }
}

# Save with metadata
joblib.dump({
    'model': model,
    'metadata': model_metadata,
    'feature_names': vectorizer.get_feature_names_out()
}, 'models/baseline_v1.0.0.pkl')
```

##### Model Registry
- Track all model versions
- Compare performance across versions
- Enable easy rollback

##### Containerization
```dockerfile
# Dockerfile
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/ /app/models/
COPY src/ /app/src/

WORKDIR /app
CMD ["python", "serve.py"]
```

#### Output Location
- `models/versions/`
- `Dockerfile`
- `docker-compose.yml`

---

### 11.8 CLI Predict Script

#### Purpose
Provide command-line interface for batch predictions.

#### Implementation Example
```python
# cli_predict.py
import argparse
import joblib
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='SMS Spam Prediction CLI')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--model', default='models/baseline/baseline_model.pkl')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Load model
    model = joblib.load(args.model)
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Predict
    predictions = model.predict(df['text'])
    probabilities = model.predict_proba(df['text'])[:, 1]
    
    # Add to dataframe
    df['prediction'] = predictions
    df['spam_probability'] = probabilities
    df['is_spam'] = probabilities >= args.threshold
    
    # Save
    df.to_csv(args.output, index=False)
    print(f"✓ Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
```

#### Usage
```bash
# Predict on batch of messages
python cli_predict.py --input messages.csv --output results.csv

# Use custom threshold
python cli_predict.py --input messages.csv --output results.csv --threshold 0.7

# Use DistilBERT model
python cli_predict.py --input messages.csv --output results.csv \
    --model models/distilbert/checkpoint-837
```

#### Output Location
`cli_predict.py`

---

### 11.9 Gradio Demo

#### Purpose
Create a web-based demo for interactive testing and stakeholder presentations.

#### Implementation Example
```python
# gradio_demo.py
import gradio as gr
import joblib

# Load model
model = joblib.load('models/baseline/baseline_model.pkl')

def predict_spam(message):
    """Predict if message is spam."""
    if not message.strip():
        return "Please enter a message", 0.5
    
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0, 1]
    
    label = "🚫 SPAM" if prediction == 1 else "✅ HAM"
    confidence = probability if prediction == 1 else (1 - probability)
    
    return f"{label} (Confidence: {confidence:.2%})", probability

# Create interface
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter SMS message here...",
        label="SMS Message"
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Slider(0, 1, label="Spam Probability")
    ],
    title="SMS Spam Classifier",
    description="Enter an SMS message to check if it's spam or legitimate.",
    examples=[
        ["Hi! How are you doing today?"],
        ["FREE entry to win! Text WIN to 12345"],
        ["Meeting at 3pm tomorrow. See you there!"],
        ["URGENT! Your account has been compromised. Click here NOW!"]
    ],
    theme="default"
)

# Launch
demo.launch(share=True)
```

#### Usage
```bash
pip install gradio
python gradio_demo.py
```

#### Features
- Real-time predictions
- Confidence visualization
- Pre-loaded examples
- Shareable link
- Mobile-friendly

#### Output Location
`gradio_demo.py`

---

### 11.10 Additional Documentation

#### API Documentation
Generate with Sphinx or MkDocs:
```bash
sphinx-quickstart docs/
make html
```

#### Model Card (Detailed)
Following Model Cards for Model Reporting framework:
- Model architecture details
- Training procedure
- Evaluation results
- Ethical considerations
- Limitations and biases
- Intended use cases

#### Deployment Guide
- Infrastructure requirements
- Scaling considerations
- Monitoring setup
- Alerting thresholds
- Update procedures

#### Output Location
- `docs/` - Generated documentation
- `MODEL_CARD.md` - Detailed model card
- `DEPLOYMENT.md` - Deployment guide

---

## Summary

### Completed Steps ✅
1. ✅ Repository and environment setup
2. ✅ Data download and exploration
3. ✅ Data cleaning and preprocessing
4. ✅ Stratified train/val/test split (80/10/10)
5. ✅ Baseline model training (TF-IDF + Naive Bayes)
6. ✅ Modern baseline (DistilBERT fine-tuning)
7. ✅ Evaluation on frozen test set
8. ✅ Comprehensive model comparison
9. ✅ Interactive testing interface
10. ✅ Documentation (README, this document)

### Future Enhancements 🔮
11. 🔲 GridSearchCV hyperparameter tuning
12. 🔲 Probability calibration and threshold optimization
13. 🔲 Error analysis and robustness checks
14. 🔲 Explainability artifacts (LIME, attention viz)
15. 🔲 Model versioning and packaging
16. 🔲 CLI prediction script
17. 🔲 Gradio web demo
18. 🔲 Detailed model card

### Key Achievements
- **Reproducibility**: Fixed random seeds, versioned dependencies
- **Best practices**: Stratified splits, comprehensive metrics, no test set leakage
- **Fair comparison**: Both models evaluated on identical test set
- **Production-ready code**: Modular, documented, error-handled
- **Clear documentation**: Multiple levels from README to detailed steps

### Files Created
| Category | Files | Location | Included in Repo? |
|----------|-------|----------|-------------------|
| Data | 3 CSV files | `data/processed/` | ✅ Yes |
| Classical Model | 3 files | `models/baseline/` | ❌ No (reproducible) |
| Transformer Model | 4+ files | `models/distilbert/` | ❌ No (reproducible) |
| Reports | 3 files | `reports/` | ✅ Yes (optional) |
| Scripts | 6 Python files | Root directory | ✅ Yes |
| Modules | 3 Python files | `src/` | ✅ Yes |
| Documentation | 3 Markdown files | Root directory | ✅ Yes |

### Total Project Size
- **Code & Docs**: ~3,500 lines (Python + Markdown)
- **Included in repo**: ~2 MB (code, data, docs)
- **Generated by training**: ~270 MB (models, not in repo)
  - Baseline model: ~500 KB
  - DistilBERT checkpoint: ~268 MB
  - Additional files: ~2 MB

> **Repository Size:** Only ~2MB of essential code and data is committed. The large model files (~270MB) are excluded and can be reproduced by running the training scripts.

---

## Conclusion

This project demonstrates a complete machine learning workflow from data ingestion to model deployment, with emphasis on:
- **Reproducibility** (fixed seeds, versioned dependencies, regenerable models)
- **Best practices** (stratified splits, comprehensive evaluation)
- **Fair comparison** (classical vs modern approaches)
- **Clear documentation** (for portfolio and collaboration)
- **Efficient storage** (code-focused repo, models generated on-demand)

The codebase is structured for easy extension with the future enhancements listed above, making it suitable for both portfolio presentation and production deployment.

### Reproducing This Project

To fully reproduce this project from the repository:

1. **Clone and setup** (~1 minute)
   ```bash
   git clone <repository-url>
   cd SMS_ham_Spam
   pip install -r requirements.txt
   ```

2. **Process data** (~5 seconds)
   ```bash
   python make_data.py
   ```

3. **Train baseline model** (~2 seconds)
   ```bash
   python train_baseline.py
   ```

4. **Train DistilBERT model** (~2 hours CPU / ~20 minutes GPU)
   ```bash
   python train_distilbert.py
   ```

5. **Compare models** (~1 minute)
   ```bash
   python compare_models.py
   ```

**Total time:** ~2 hours on CPU, or ~30 minutes with GPU access.
**Storage required:** ~300 MB for generated models and data.

