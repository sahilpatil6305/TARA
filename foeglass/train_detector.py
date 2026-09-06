import glob
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.audio_features import extract_mfcc_mean_std


def extract_features(audio_path):
    try:
        return extract_mfcc_mean_std(audio_path, sr=16000, n_mfcc=13)
    except Exception as exc:
        print(f"[!] Error reading {audio_path}: {exc}")
        return None


def load_training_data():
    real_dir = "dataset/train/real"
    fake_dir = "dataset/train/fake"

    features = []
    labels = []

    print("[*] Loading real training dataset...")
    for path in glob.glob(os.path.join(real_dir, "*.wav")):
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            labels.append("real")

    print("[*] Loading fake training dataset...")
    for path in glob.glob(os.path.join(fake_dir, "*.wav")):
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            labels.append("fake")

    if not features:
        raise RuntimeError(
            "[!] No training data found. dataset/train/real and dataset/train/fake must contain .wav files."
        )

    return np.array(features), np.array(labels)


def _save_training_validation(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_validation.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(f"[*] Saved training validation summary to {output_path}")


def _save_confusion_matrix_plot(matrix, plot_path):
    plt.figure(figsize=(5, 4))
    image = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(image)
    labels = ["Fake", "Real"]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Training Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(int(matrix[row_idx, col_idx])),
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def _save_metric_plot(metrics, plot_path):
    labels = list(metrics.keys())
    values = [float(metrics[label]) for label in labels]
    positions = np.arange(len(labels))

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(positions, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    plt.xticks(positions, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Training Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def train(output_dir="results"):
    print("[*] === Training Detector ===")
    X, y = load_training_data()
    y_binary = (y == "real").astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    cv_scores = cross_val_score(cv_model, X, y_binary, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"[*] CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("[*] Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=["fake", "real"])
    y_pred_binary = (preds == "real").astype(int)
    probs = clf.predict_proba(X)
    real_idx = list(clf.classes_).index("real")
    real_probs = probs[:, real_idx]
    roc = roc_auc_score(y_binary, real_probs)
    precision = precision_score(y_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_binary, y_pred_binary, zero_division=0)
    prob_true, prob_pred = calibration_curve(y_binary, real_probs, n_bins=10, strategy="uniform")

    print(f"[*] Accuracy on training set: {acc:.4f}")
    print(f"[*] Precision on training set: {precision:.4f}")
    print(f"[*] Recall on training set: {recall:.4f}")
    print(f"[*] ROC-AUC: {roc:.4f}")
    print(f"[*] Confusion Matrix:\n{cm}")

    os.makedirs("models", exist_ok=True)
    model_path = "models/detector.pkl"
    joblib.dump(clf, model_path)
    print(f"[*] Saved model to {model_path}")

    results = {
        "feature_design": "mfcc_mean_plus_std",
        "feature_dimension": int(X.shape[1]),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "train_accuracy": float(acc),
        "train_precision": float(precision),
        "train_recall": float(recall),
        "train_roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "calibration_curve": {
            "prob_true": [float(value) for value in prob_true],
            "prob_pred": [float(value) for value in prob_pred],
        },
    }
    _save_training_validation(results, output_dir=output_dir)

    _save_confusion_matrix_plot(cm, os.path.join(output_dir, "confusion_matrix.png"))
    _save_metric_plot(
        {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "ROC-AUC": roc,
        },
        os.path.join(output_dir, "training_metrics.png"),
    )
    return results


if __name__ == "__main__":
    train()
