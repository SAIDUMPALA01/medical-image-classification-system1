"""
evaluate.py
-----------
Full evaluation script for the Pneumonia Detection CNN.
Generates confusion matrix, classification report, and ROC-AUC score.

Usage:
    python src/evaluate.py --data-dir data/ --model-path models/pneumonia_detector.h5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model

from data_loader import get_data_generators


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Pneumonia Detection CNN")
    parser.add_argument("--data-dir",    type=str, default="data/",                       help="Root data directory")
    parser.add_argument("--model-path",  type=str, default="models/pneumonia_detector.h5",help="Path to trained model")
    parser.add_argument("--batch-size",  type=int, default=32,                            help="Batch size")
    parser.add_argument("--output-dir",  type=str, default="images/",                     help="Directory to save output plots")
    return parser.parse_args()


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {path}")


# ── ROC Curve Plot ────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_scores, output_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {path}")
    return auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    # Load test data
    _, _, test_gen = get_data_generators(args.data_dir, args.batch_size)
    class_names = list(test_gen.class_indices.keys())
    print(f"Classes: {class_names}")
    print(f"Test samples: {test_gen.samples}\n")

    # Predictions
    test_gen.reset()
    y_scores = model.predict(test_gen, verbose=1).flatten()
    y_pred   = (y_scores >= 0.5).astype(int)
    y_true   = test_gen.classes

    # Metrics
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT — TEST SET")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    auc = plot_roc_curve(y_true, y_scores, args.output_dir)
    print(f"\nAUC-ROC Score: {auc:.4f}")

    # Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, args.output_dir)

    # Summary
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"True Negatives  (Normal   → Normal):    {tn}")
    print(f"False Positives (Normal   → Pneumonia): {fp}")
    print(f"False Negatives (Pneumonia→ Normal):    {fn}")
    print(f"True Positives  (Pneumonia→ Pneumonia): {tp}")
    print(f"Sensitivity (Recall): {tp / (tp + fn):.4f}")
    print(f"Specificity:          {tn / (tn + fp):.4f}")


if __name__ == "__main__":
    main()
