"""
predict.py
----------
Inference script for the Pneumonia Detection CNN.
Predicts Normal or Pneumonia from a single chest X-ray image.

Usage:
    python src/predict.py --image-path path/to/xray.jpg
    python src/predict.py --image-path path/to/xray.jpg --model-path models/pneumonia_detector.h5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from data_loader import load_single_image


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray")
    parser.add_argument("--image-path",  type=str, required=True,                         help="Path to the X-ray image")
    parser.add_argument("--model-path",  type=str, default="models/pneumonia_detector.h5",help="Path to trained model")
    parser.add_argument("--show-image",  action="store_true",                              help="Display the image with prediction")
    return parser.parse_args()


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image_path: str, model_path: str):
    """
    Run inference on a single chest X-ray image.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the trained .h5 model.

    Returns:
        dict: prediction label, confidence, and raw score.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)
    img   = load_single_image(image_path)

    score = model.predict(img, verbose=0)[0][0]
    label = "PNEUMONIA" if score >= 0.5 else "NORMAL"
    confidence = score if score >= 0.5 else 1 - score

    return {"label": label, "confidence": float(confidence), "raw_score": float(score)}


# ── Display ───────────────────────────────────────────────────────────────────
def show_prediction(image_path: str, result: dict):
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(image_path, target_size=(224, 224))
    color = "red" if result["label"] == "PNEUMONIA" else "green"

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Prediction: {result['label']}\nConfidence: {result['confidence']:.2%}",
        fontsize=14,
        fontweight="bold",
        color=color,
    )
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\nAnalyzing: {args.image_path}")
    print(f"Using model: {args.model_path}\n")

    result = predict(args.image_path, args.model_path)

    print("=" * 40)
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  Raw Score  : {result['raw_score']:.4f}  (threshold: 0.5)")
    print("=" * 40)

    if result["label"] == "PNEUMONIA":
        print("\n⚠️  Pneumonia detected. Please consult a qualified healthcare professional.")
    else:
        print("\n✅  No pneumonia detected.")

    print("\n⚠️  DISCLAIMER: This tool is for research purposes only.")
    print("   Do not use for clinical diagnosis without professional validation.\n")

    if args.show_image:
        show_prediction(args.image_path, result)


if __name__ == "__main__":
    main()
