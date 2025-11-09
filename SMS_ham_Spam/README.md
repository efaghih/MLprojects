# SMS Spam Classification Project

A machine learning project to classify SMS messages as spam or ham (legitimate).

## 📁 Project Structure

```
SMS_ham_Spam/
├── Dataset/
│   └── SMSSpamCollection       # Raw dataset (5,572 messages)
├── data/
│   └── processed/              # Processed and split data
│       ├── train.csv           # Training set (80% - 4,457 samples)
│       ├── val.csv             # Validation set (10% - 557 samples)
│       └── test.csv            # Test set (10% - 558 samples)
├── models/                     # ⚠️ NOT INCLUDED IN REPO (see note below)
│   ├── baseline/               # Classical ML model
│   │   ├── baseline_model.pkl  # Trained TF-IDF + Naive Bayes
│   │   ├── metrics.txt         # Performance metrics
│   │   └── confusion_matrices.png
│   └── distilbert/             # Transformer model
│       ├── checkpoint-837/     # Best model checkpoint
│       ├── metrics.txt         # Performance metrics
│       └── confusion_matrices.png
├── reports/                    # Model comparison reports
│   ├── model_comparison.txt    # Detailed comparison
│   ├── comparison_metrics.csv  # Metrics table
│   └── model_comparison.png    # Visualizations
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data.py                 # Data ingestion and preprocessing
│   ├── train_baseline.py       # Baseline model training
│   └── train_distilbert.py     # DistilBERT training
├── make_data.py                # Data processing script
├── train_baseline.py           # Train baseline model
├── train_distilbert.py         # Train DistilBERT model
├── compare_models.py           # Compare both models
├── test_model.py               # Interactive model testing
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

> **⚠️ Important Note:** The `models/` folder and trained model files are **NOT included in this repository** because they exceed 1GB in size (primarily due to the DistilBERT checkpoint which is ~268MB). You can easily reproduce all models by running the training scripts as described below.

## 📊 Dataset Information

- **Total Messages**: 5,572
- **Ham (Legitimate)**: 4,825 messages (86.59%)
- **Spam**: 747 messages (13.41%)
- **Data Split**: Stratified 80/10/10 (train/val/test)

All splits maintain the same spam/ham ratio (~13.4% spam).

### Dataset Citation

This project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository:

**Citation:**
> Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

**Source:** [UCI ML Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Original Paper:**
> Almeida, T.A., Hidalgo, J.M.G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. In Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG '11).

## 🚀 Usage

### Step 1: Data Ingestion (COMPLETED ✅)

To process the raw data and create train/val/test splits:

```bash
python make_data.py
```

This will:
1. Load the raw SMS Spam Collection data
2. Create binary labels (0=ham, 1=spam)
3. Perform stratified train/val/test split (80/10/10)
4. Save the processed CSV files to `data/processed/`

### Step 2: Baseline Model Training (COMPLETED ✅)

> **📝 Note:** Since trained models are not included in the repository (size > 1GB), you need to train them first by running the script below.

To train the baseline TF-IDF + Naive Bayes model:

```bash
python train_baseline.py
```

**Training time:** ~2 seconds on a standard CPU

This will:
1. Load training and validation data
2. Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes
3. Train the model on training set
4. Evaluate on both training and validation sets
5. Generate confusion matrices and save the model

**Baseline Results (Test Set):**
- **F1 Score (Spam)**: 0.9209
- **ROC-AUC**: 0.9944
- **PR-AUC**: 0.9810
- **Accuracy**: 98.03%
- **False Positives**: 0 (Perfect! No legitimate messages blocked)
- **False Negatives**: 11 (Some spam slips through)

Output files:
- `models/baseline/baseline_model.pkl` - Trained model
- `models/baseline/metrics.txt` - Performance metrics
- `models/baseline/confusion_matrices.png` - Visualizations

### Step 3: DistilBERT Model (Modern Baseline) (COMPLETED ✅)

> **📝 Note:** The DistilBERT model checkpoint (~268MB) is not included in the repository. Run the training script below to generate it.

To train the DistilBERT transformer model:

```bash
python train_distilbert.py
```

**Training time:** ~2 hours on CPU, ~20 minutes on GPU

This will:
1. Load and tokenize the data (max_length=128)
2. Fine-tune DistilBERT (distilbert-base-uncased)
3. Train for 3 epochs with batch_size=16, lr=2e-5
4. Evaluate on train/validation/test sets
5. Save model checkpoints and metrics

**DistilBERT Results (Test Set):**
- **F1 Score (Spam)**: 0.9664
- **ROC-AUC**: 0.9935
- **PR-AUC**: 0.9880
- **Accuracy**: 99.10%
- **False Positives**: 2
- **False Negatives**: 3

Output files:
- `models/distilbert/checkpoint-837/` - Best model checkpoint
- `models/distilbert/metrics.txt` - Performance metrics
- `models/distilbert/confusion_matrices.png` - Visualizations

### Step 4: Model Comparison (COMPLETED ✅)

> **📝 Note:** This step requires both models to be trained first (Steps 2 & 3).

To compare both models on the test set:

```bash
python compare_models.py
```

**Comparison Results:**

| Metric | TF-IDF + NB | DistilBERT | Improvement |
|--------|-------------|------------|-------------|
| **F1 Score** | 0.9209 | **0.9664** | +4.56 pp |
| **Accuracy** | 0.9803 | **0.9910** | +1.10 pp |
| **Recall** | 0.8533 | **0.9600** | +12.5% |
| **Precision** | **1.0000** | 0.9730 | -2.70% |
| **ROC-AUC** | **0.9944** | 0.9935 | -0.09 pp |

**Key Insights:**
- ✅ **DistilBERT catches 8 more spam messages** (3 vs 11 false negatives)
- ⚠️ **Trade-off: 2 false positives** (baseline had 0)
- 🎯 **Overall better performance** with +4.56pp F1 improvement
- ⚡ **Training time**: ~2 hours (CPU) vs 2 seconds (baseline)

Output files:
- `reports/model_comparison.txt` - Detailed comparison
- `reports/comparison_metrics.csv` - Metrics table
- `reports/model_comparison.png` - Visualization

### Next Steps

- Deploy the chosen model (baseline for speed, DistilBERT for accuracy)
- Add model serving with FastAPI
- Create web interface

## 🔧 Requirements

### Core Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn

### Deep Learning (for DistilBERT)
- torch
- transformers
- datasets
- evaluate
- accelerate

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Project Highlights

### Technical Achievements
✅ **End-to-end ML pipeline** from data ingestion to model comparison
✅ **Classical ML baseline** (TF-IDF + Naive Bayes) - Fast and interpretable
✅ **Modern transformer baseline** (DistilBERT) - State-of-the-art accuracy
✅ **Comprehensive evaluation** on fixed test set with multiple metrics
✅ **Fair comparison** between classical and deep learning approaches
✅ **Production-ready code** with proper documentation and error handling
✅ **Fully reproducible** - All models can be regenerated from source code

### Key Learnings
- **Classical ML** still performs excellently on text classification (98% accuracy)
- **Transformers** provide incremental improvements but at higher computational cost
- **Trade-offs matter**: Baseline has zero false positives, DistilBERT has better recall
- **Proper evaluation** with stratified splits and multiple metrics is crucial

### Best Practices Demonstrated
- Reproducible experiments (fixed random seeds)
- Stratified splitting for imbalanced data
- Comprehensive metrics (F1, ROC-AUC, PR-AUC, confusion matrices)
- Well-commented, modular code
- Clear documentation and comparison reports

## 🎯 Model Selection Guide

**Choose TF-IDF + Naive Bayes if:**
- ✅ You need fast inference (< 1ms per message)
- ✅ You want zero false positives
- ✅ You need an interpretable model
- ✅ You have limited compute resources

**Choose DistilBERT if:**
- ✅ You want maximum spam detection (96% recall)
- ✅ You can tolerate occasional false positives
- ✅ You have GPU available for inference
- ✅ You want state-of-the-art performance

## 📊 Summary Statistics

| Aspect | Value |
|--------|-------|
| **Dataset Size** | 5,572 messages |
| **Spam Ratio** | 13.41% |
| **Training Time (Baseline)** | ~2 seconds |
| **Training Time (DistilBERT)** | ~2 hours (CPU) |
| **Best F1 Score** | 0.9664 (DistilBERT) |
| **Best Precision** | 1.0000 (Baseline) |
| **Best Recall** | 0.9600 (DistilBERT) |

## 📝 Notes

- Random state is fixed at 42 for reproducibility
- Stratified splitting ensures balanced class distribution across all splits
- Both models evaluated on the same fixed test set for fair comparison
- Confusion matrices and detailed reports available in respective directories

