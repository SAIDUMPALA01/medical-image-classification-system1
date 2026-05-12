"""
train.py
--------
Training script for the Pneumonia Detection CNN.

Usage:
    python src/train.py --data-dir data/ --epochs 50 --batch-size 32 --learning-rate 0.001
"""

import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_model
from data_loader import get_data_generators


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Pneumonia Detection CNN")
    parser.add_argument("--data-dir",      type=str,   default="data/",  help="Root data directory")
    parser.add_argument("--epochs",        type=int,   default=50,       help="Number of training epochs")
    parser.add_argument("--batch-size",    type=int,   default=32,       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,    help="Initial learning rate")
    parser.add_argument("--output-dir",    type=str,   default="models/",help="Directory to save model weights")
    return parser.parse_args()


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "pneumonia_detector.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ── Plot & Save Training History ──────────────────────────────────────────────
def save_training_plots(history, output_dir: str = "images/"):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training plots saved to {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    # Data
    train_gen, val_gen, test_gen = get_data_generators(args.data_dir, args.batch_size)
    print(f"\nClass indices: {train_gen.class_indices}")
    print(f"Training samples:   {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples:       {test_gen.samples}\n")

    # Model
    model = build_model(learning_rate=args.learning_rate)
    model.summary()

    # Train
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(args.output_dir),
        verbose=1,
    )

    # Save plots
    save_training_plots(history)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print(f"\nBest model saved to {args.output_dir}/pneumonia_detector.h5")


if __name__ == "__main__":
    main()
