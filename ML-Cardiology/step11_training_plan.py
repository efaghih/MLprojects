"""
Step 11: Training and Evaluation Plan
Purpose: Outline the complete training and evaluation workflow
"""

print("=" * 60)
print("COMPLETE TRAINING & EVALUATION WORKFLOW")
print("=" * 60)

workflow = [
    "1. Load all WAV files and create clips + spectrograms",
    "2. Assign labels: each clip gets patient's Outcome",
    "3. Split patients into 5 folds (patient-level)",
    "4. For each fold:",
    "   a. Train CNN on training patients' clips",
    "   b. Predict on test patients' clips",
    "   c. Aggregate clip predictions -> patient predictions",
    "   d. Calculate patient-level metrics",
    "5. Average metrics across all 5 folds",
    "6. Report final accuracy, ROC-AUC, F1, Sensitivity, Specificity"
]

print("\nTraining Steps:")
for step in workflow:
    print(f"  {step}")

print("\n" + "=" * 60)
print("KEY POINTS:")
print("=" * 60)
print("  - Train on CLIPS, but evaluate on PATIENTS")
print("  - Aggregate: mean of clip probabilities -> patient score")
print("  - Metrics: ROC-AUC, F1, Sensitivity, Specificity")
print("  - Cross-validation: 5 folds, patient-level splits")
print("  - Class imbalance: Use class weights if needed")

print("\n[OK] Step 11 complete: Training plan ready")
print("\nNext: Implement full training pipeline with TensorFlow/Keras")

