import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.detector import Detector


def _save_confusion_matrix_plot(matrix, labels, plot_path, title):
    plt.figure(figsize=(5, 4))
    image = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(image)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

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


def _save_metric_bar_chart(metric_map, plot_path, title):
    labels = list(metric_map.keys())
    values = [float(metric_map[label]) for label in labels]
    positions = np.arange(len(labels))

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(positions, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"][: len(labels)])
    plt.ylim(0, 1.05)
    plt.xticks(positions, labels)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def _save_precision_recall_curve_plot(precision_values, recall_values, average_precision, plot_path):
    plt.figure(figsize=(6, 5))
    plt.plot(recall_values, precision_values, color="#1f77b4", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={average_precision:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def _save_asr_success_plot(rows, plot_path, threshold):
    iterations = [row["iteration"] for row in rows]
    scores = [row["score"] for row in rows]
    success_flags = [1 if row["attack_success"] else 0 for row in rows]
    colors = ["#2ca02c" if flag else "#d62728" for flag in success_flags]

    fig, axis_score = plt.subplots(figsize=(10, 4.8))
    axis_score.plot(iterations, scores, marker="o", color="#1f77b4", linewidth=2, label="Detector score")
    axis_score.axhline(threshold, linestyle="--", color="gray", label=f"ASR threshold ({threshold:.1f})")
    axis_score.set_xlabel("Iteration")
    axis_score.set_ylabel("Detector Score")
    axis_score.set_ylim(0, 1.05)
    axis_score.grid(True, linestyle="--", alpha=0.35)

    axis_success = axis_score.twinx()
    axis_success.bar(iterations, success_flags, alpha=0.25, color=colors, label="Attack success")
    axis_success.set_ylabel("Attack Success")
    axis_success.set_ylim(0, 1.1)
    axis_success.set_yticks([0, 1])
    axis_success.set_yticklabels(["Fail", "Success"])

    handles_a, labels_a = axis_score.get_legend_handles_labels()
    handles_b, labels_b = axis_success.get_legend_handles_labels()
    axis_score.legend(handles_a + handles_b, labels_a + labels_b, loc="lower right")
    plt.title("ASR Success by Iteration")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)


def _save_cumulative_asr_plot(rows, plot_path):
    iterations = [row["iteration"] for row in rows]
    cumulative_successes = np.cumsum([1 if row["attack_success"] else 0 for row in rows])
    cumulative_asr = (cumulative_successes / np.arange(1, len(rows) + 1)) * 100.0

    plt.figure(figsize=(8, 4.5))
    plt.plot(iterations, cumulative_asr, marker="o", color="#2ca02c", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative ASR (%)")
    plt.ylim(0, 105)
    plt.title("Cumulative Attack Success Rate")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def compute_generated_metrics(run_audio_dir, success_threshold=0.5):
    rows = []

    for root, _, files in os.walk(run_audio_dir):
        if "metadata.json" not in files:
            continue

        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        rows.append(
            {
                "iteration": int(data["iteration"]),
                "score": float(data.get("score", data["real_score"])),
                "distance": None if data.get("distance") is None else float(data["distance"]),
                "objective": float(data.get("objective", data["final_score"])),
                "attack_success": bool(float(data.get("score", data["real_score"])) >= success_threshold),
                "final_score": float(data["final_score"]),
            }
        )

    rows.sort(key=lambda row: row["iteration"])
    if not rows:
        raise RuntimeError("[!] No generated audio metadata found in data/audio.")

    total = len(rows)
    avg_score = sum(row["score"] for row in rows) / total
    avg_objective = sum(row["objective"] for row in rows) / total
    avg_final_score = sum(row["final_score"] for row in rows) / total
    classified_real = sum(1 for row in rows if row["attack_success"])
    asr = (classified_real / total) * 100.0

    return {
        "rows": rows,
        "avg_score": avg_score,
        "avg_objective": avg_objective,
        "avg_final_score": avg_final_score,
        "asr": asr,
        "classified_real": classified_real,
        "total": total,
        "threshold": success_threshold,
    }


def _pearson(values_a, values_b):
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    if np.std(values_a) == 0 or np.std(values_b) == 0:
        return 0.0
    return float(np.corrcoef(values_a, values_b)[0, 1])


def _save_feature_heatmap(correlations, plot_path):
    labels = [row["feature"] for row in correlations]
    values = np.array([[row["correlation_with_real_score"] for row in correlations]])

    plt.figure(figsize=(max(10, len(labels) * 0.4), 2.8))
    image = plt.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, label="Correlation")
    plt.yticks([0], ["real_score"])
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def analyze_run_features(run_audio_dir, run_results_dir):
    rows = []
    for root, _, files in os.walk(run_audio_dir):
        if "features.json" not in files or "metadata.json" not in files:
            continue

        with open(os.path.join(root, "features.json"), "r", encoding="utf-8") as handle:
            features = json.load(handle)
        with open(os.path.join(root, "metadata.json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        rows.append(
            {
                "iteration": int(metadata["iteration"]),
                "real_score": float(metadata["real_score"]),
                "prompt": metadata["prompt"],
                "feature_names": features["feature_names"],
                "feature_values": [float(value) for value in features["feature_values"]],
            }
        )

    rows.sort(key=lambda row: row["iteration"])
    if not rows:
        raise RuntimeError("[!] No feature logs found for generated audio run.")

    feature_names = rows[0]["feature_names"]
    scores = [row["real_score"] for row in rows]
    feature_columns = list(zip(*[row["feature_values"] for row in rows]))

    correlations = []
    for name, values in zip(feature_names, feature_columns):
        correlations.append(
            {
                "feature": name,
                "correlation_with_real_score": _pearson(values, scores),
                "high_score_mean": float(
                    np.mean([value for value, score in zip(values, scores) if score >= np.median(scores)])
                ),
                "low_score_mean": float(
                    np.mean([value for value, score in zip(values, scores) if score < np.median(scores)])
                ),
            }
        )

    correlations.sort(key=lambda item: abs(item["correlation_with_real_score"]), reverse=True)
    top_prompts = sorted(rows, key=lambda row: row["real_score"], reverse=True)[:5]

    summary = {
        "top_correlated_features": correlations[:5],
        "all_feature_correlations": correlations,
        "top_prompts_by_real_score": [
            {
                "iteration": row["iteration"],
                "prompt": row["prompt"],
                "real_score": row["real_score"],
            }
            for row in top_prompts
        ],
        "mean_real_score": float(np.mean(scores)),
        "max_real_score": float(np.max(scores)),
    }

    json_path = os.path.join(run_results_dir, "feature_analysis.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    csv_path = os.path.join(run_results_dir, "feature_correlations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "correlation_with_real_score", "high_score_mean", "low_score_mean"])
        for row in correlations:
            writer.writerow(
                [
                    row["feature"],
                    row["correlation_with_real_score"],
                    row["high_score_mean"],
                    row["low_score_mean"],
                ]
            )

    plots_dir = os.path.join(run_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    heatmap_path = os.path.join(plots_dir, "feature_correlation_heatmap.png")
    _save_feature_heatmap(correlations, heatmap_path)

    print(f"[*] Feature analysis saved to {json_path}")
    print(f"[*] Feature correlations saved to {csv_path}")
    print(f"[*] Feature heatmap saved to {heatmap_path}")
    return summary


def evaluate_eval_set(run_results_dir=None, validation_dir="results"):
    """
    Evaluate the detector only on dataset/eval and compute FNR = FN / (FN + TP).
    """
    print("\n[*] === Evaluating on mini-RITW Eval Set ===")
    detector = Detector()

    real_dir = "dataset/eval/real"
    fake_dir = "dataset/eval/fake"

    y_true = []
    y_scores = []
    preds = []

    print("[*] Scoring eval real samples...")
    for path in glob.glob(os.path.join(real_dir, "*.wav")):
        score = detector.predict_real_score(path)
        y_true.append(1)
        y_scores.append(score)
        preds.append(1 if score >= 0.5 else 0)

    print("[*] Scoring eval fake samples...")
    for path in glob.glob(os.path.join(fake_dir, "*.wav")):
        score = detector.predict_real_score(path)
        y_true.append(0)
        y_scores.append(score)
        preds.append(1 if score >= 0.5 else 0)

    if not y_true:
        raise RuntimeError("[!] No eval data found in dataset/eval.")

    acc = accuracy_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    matrix = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="uniform")
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    print(f"[*] Accuracy: {acc:.4f}")
    print(f"[*] Precision: {precision:.4f}")
    print(f"[*] Recall: {recall:.4f}")
    print(f"[*] ROC-AUC: {roc_auc:.4f}")
    print(f"[*] Confusion Matrix:\n{matrix}")
    print(f"[*] True Positives (Real as Real): {tp}")
    print(f"[*] True Negatives (Fake as Fake): {tn}")
    print(f"[*] False Positives (Fake as Real): {fp}")
    print(f"[*] False Negatives (Real as Fake): {fn}")
    print(f"[*] FNR (eval set): {fnr:.4f}")

    os.makedirs(validation_dir, exist_ok=True)
    global_results_path = os.path.join(validation_dir, "eval_set_results.csv")
    with open(global_results_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Accuracy", "Precision", "Recall", "ROC_AUC", "TP", "TN", "FP", "FN", "FNR"])
        writer.writerow([acc, precision, recall, roc_auc, tp, tn, fp, fn, fnr])

    calibration_payload = {
        "prob_true": [float(value) for value in prob_true],
        "prob_pred": [float(value) for value in prob_pred],
    }
    calibration_path = os.path.join(validation_dir, "calibration_curve.json")
    with open(calibration_path, "w", encoding="utf-8") as handle:
        json.dump(calibration_payload, handle, indent=4)

    classifier_metrics_payload = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": matrix.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fnr": float(fnr),
        "precision_recall_curve": {
            "precision": [float(value) for value in pr_precision],
            "recall": [float(value) for value in pr_recall],
        },
    }
    classifier_metrics_path = os.path.join(validation_dir, "classifier_metrics.json")
    with open(classifier_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(classifier_metrics_payload, handle, indent=4)

    if run_results_dir:
        run_fnr_path = os.path.join(run_results_dir, "fnr_results.csv")
        with open(run_fnr_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Accuracy", "Precision", "Recall", "ROC_AUC", "TP", "TN", "FP", "FN", "FNR"])
            writer.writerow([acc, precision, recall, roc_auc, tp, tn, fp, fn, fnr])

        plots_dir = os.path.join(run_results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        calibration_plot_path = os.path.join(plots_dir, "calibration_curve.png")
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        plt.plot(prob_pred, prob_true, marker="o", label="Detector")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(calibration_plot_path)
        plt.close()

        run_calibration_path = os.path.join(run_results_dir, "calibration_curve.json")
        with open(run_calibration_path, "w", encoding="utf-8") as handle:
            json.dump(calibration_payload, handle, indent=4)

        confusion_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
        _save_confusion_matrix_plot(matrix, ["Fake", "Real"], confusion_plot_path, "Eval Set Confusion Matrix")

        metric_bar_path = os.path.join(plots_dir, "classification_metrics.png")
        _save_metric_bar_chart(
            {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "ROC-AUC": roc_auc,
                "Avg Precision": average_precision,
            },
            metric_bar_path,
            "Eval Set Classification Metrics",
        )

        precision_recall_plot_path = os.path.join(plots_dir, "precision_recall_curve.png")
        _save_precision_recall_curve_plot(pr_precision, pr_recall, average_precision, precision_recall_plot_path)

        run_classifier_metrics_path = os.path.join(run_results_dir, "classifier_metrics.json")
        with open(run_classifier_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(classifier_metrics_payload, handle, indent=4)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fnr": fnr,
        "calibration_curve": calibration_payload,
    }


def evaluate_generated(run_audio_dir, run_results_dir):
    """
    Evaluate generated audio only from data/audio and compute ASR using real_score >= 0.5.
    """
    print(f"\n[*] === Evaluating Generated Audio ({os.path.basename(run_audio_dir)}) ===")

    metrics = compute_generated_metrics(run_audio_dir, success_threshold=0.5)
    rows = metrics["rows"]
    total = metrics["total"]
    avg_score = metrics["avg_score"]
    avg_objective = metrics["avg_objective"]
    avg_final_score = metrics["avg_final_score"]
    asr = metrics["asr"]

    print(f"[*] Total Generated Samples: {total}")
    print(f"[*] Average Detector Score: {avg_score:.4f}")
    print(f"[*] Average Objective: {avg_objective:.4f}")
    print(f"[*] Average Final Score: {avg_final_score:.4f}")
    print(f"[*] Attack Success Rate (ASR): {asr:.2f}%")

    output_path = os.path.join(run_results_dir, "generated_audio_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "score", "distance", "objective", "final_score", "attack_success"])
        for row in rows:
            writer.writerow(
                [row["iteration"], row["score"], row["distance"], row["objective"], row["final_score"], row["attack_success"]]
            )

    plots_dir = os.path.join(run_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _save_asr_success_plot(rows, os.path.join(plots_dir, "asr_success_by_iteration.png"), threshold=0.5)
    _save_cumulative_asr_plot(rows, os.path.join(plots_dir, "cumulative_asr.png"))

    generated_metrics_path = os.path.join(run_results_dir, "generated_metrics.json")
    with open(generated_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "avg_score": float(avg_score),
                "avg_objective": float(avg_objective),
                "avg_final_score": float(avg_final_score),
                "asr": float(asr),
                "classified_real": int(metrics["classified_real"]),
                "total": int(total),
                "threshold": 0.5,
            },
            handle,
            indent=4,
        )

    return {
        "rows": rows,
        "avg_score": avg_score,
        "avg_objective": avg_objective,
        "avg_final_score": avg_final_score,
        "asr": asr,
        "classified_real": metrics["classified_real"],
        "total": total,
    }


def compare_generated_thresholds(run_audio_dir, run_results_dir, base_threshold=0.5, strict_threshold=0.8):
    base_metrics = compute_generated_metrics(run_audio_dir, success_threshold=base_threshold)
    strict_metrics = compute_generated_metrics(run_audio_dir, success_threshold=strict_threshold)

    comparison = {
        "asr_base_threshold": base_threshold,
        "asr_base_value": base_metrics["asr"],
        "asr_strict_threshold": strict_threshold,
        "asr_strict_value": strict_metrics["asr"],
        "old_asr_at_0_5": base_metrics["asr"],
        "new_asr_at_0_8": strict_metrics["asr"],
    }

    output_path = os.path.join(run_results_dir, "asr_threshold_comparison.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=4)

    plots_dir = os.path.join(run_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    threshold_plot_path = os.path.join(plots_dir, "asr_threshold_comparison.png")
    _save_metric_bar_chart(
        {
            f"ASR@{base_threshold:.1f}": base_metrics["asr"] / 100.0,
            f"ASR@{strict_threshold:.1f}": strict_metrics["asr"] / 100.0,
        },
        threshold_plot_path,
        "ASR Threshold Comparison",
    )

    print(
        f"[*] ASR comparison saved to {output_path} "
        f"(ASR@{base_threshold:.1f}={base_metrics['asr']:.2f}%, "
        f"ASR@{strict_threshold:.1f}={strict_metrics['asr']:.2f}%)"
    )

    return comparison


def save_baseline_comparison(baseline_metrics, optimized_metrics, output_path):
    payload = {
        "baseline_asr": baseline_metrics["asr"],
        "optimized_asr": optimized_metrics["asr"],
        "baseline_avg_score": baseline_metrics["avg_final_score"],
        "optimized_avg_score": optimized_metrics["avg_final_score"],
        "asr_improvement": optimized_metrics["asr"] - baseline_metrics["asr"],
        "avg_score_improvement": optimized_metrics["avg_final_score"] - baseline_metrics["avg_final_score"],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    print(f"[*] Baseline comparison saved to {output_path}")
    return payload


def aggregate_experiment_results(baseline_runs, optimized_runs, output_path):
    def _metric_summary(rows, key):
        values = [row[key] for row in rows]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    aggregate = {
        "baseline": {
            "best_score": _metric_summary(baseline_runs, "best_score"),
            "avg_score": _metric_summary(baseline_runs, "avg_score"),
            "asr": _metric_summary(baseline_runs, "asr"),
        },
        "optimized": {
            "best_score": _metric_summary(optimized_runs, "best_score"),
            "avg_score": _metric_summary(optimized_runs, "avg_score"),
            "asr": _metric_summary(optimized_runs, "asr"),
        },
        "per_run": {
            "baseline": baseline_runs,
            "optimized": optimized_runs,
        },
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=4)
    print(f"[*] Aggregate results saved to {output_path}")
    return aggregate
