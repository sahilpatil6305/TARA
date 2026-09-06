"""
train_detector.py
=================
Step 2 of the DRAO execution sequence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import get_config
from src.detector import DetectorTrainingResult, train_detector
from src.logging_utils import setup_logger


def _bootstrap_fake(real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create synthetic generated embeddings when none exist yet."""

    sigma = 0.3 * float(real.std())
    noise = rng.normal(0.0, sigma, size=real.shape).astype(np.float32)
    return real + noise


def _save_confusion_matrix_plot(matrix: list[list[int]], target: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Detector Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Fake", "Real"])
    ax.set_yticks([0, 1], labels=["Fake", "Real"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150)
    plt.close(fig)


def _save_detector_metrics(
    result: DetectorTrainingResult,
    metrics_path: Path,
    plot_path: Path,
) -> dict[str, object]:
    payload = {
        "accuracy": result.metrics["accuracy"],
        "precision": result.metrics["precision"],
        "recall": result.metrics["recall"],
        "f1_score": result.metrics["f1_score"],
        "roc_auc": result.metrics["roc_auc"],
        "confusion_matrix": result.metrics["confusion_matrix"],
        "raw_sample_counts": {
            "real": result.train_real_count,
            "fake": result.train_fake_count,
        },
        "balanced_sample_counts": {
            "real": result.balanced_real_count,
            "fake": result.balanced_fake_count,
        },
        "train_size": result.train_count,
        "validation_size": result.eval_count,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _save_confusion_matrix_plot(result.metrics["confusion_matrix"], plot_path)
    return payload


def _save_dataset_stats(
    result: DetectorTrainingResult,
    target: Path,
) -> dict[str, object]:
    payload = {
        "total_real_samples": result.train_real_count,
        "total_fake_samples": result.train_fake_count,
        "train_count": result.train_count,
        "eval_count": result.eval_count,
        "class_balance_ratio": result.class_balance_ratio,
        "train_indices": result.train_indices,
        "eval_indices": result.eval_indices,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main() -> None:
    cfg = get_config()
    logger = setup_logger("foeglass.train_detector", cfg.paths.logs_dir / "train_detector.log")
    np.random.seed(cfg.optimization.seed)
    torch.manual_seed(cfg.optimization.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.optimization.seed)
    rng = np.random.default_rng(cfg.optimization.seed)

    real_path: Path = cfg.paths.real_embeddings_path
    if not real_path.exists():
        logger.error(
            f"[ERROR] Real embeddings not found at:\n  {real_path}\n\n"
            "Run `python build_real_embeddings.py` first."
        )
        sys.exit(1)
    real_embs = np.load(str(real_path)).astype(np.float32)
    logger.info("[train_detector] Real embeddings loaded: %s", real_embs.shape)

    gen_path: Path = cfg.paths.generated_embeddings_path
    if gen_path.exists():
        gen_embs = np.load(str(gen_path)).astype(np.float32)
        logger.info("[train_detector] Generated embeddings loaded: %s", gen_embs.shape)
    else:
        gen_embs = _bootstrap_fake(real_embs, rng)
        logger.warning(
            "[train_detector] No generated embeddings found - bootstrapped %d synthetic fakes from real distribution.",
            gen_embs.shape[0],
        )

    logger.info(
        "[train_detector] Raw sample counts | real=%d fake=%d",
        len(real_embs),
        len(gen_embs),
    )

    mc = cfg.model
    result = train_detector(
        real_embs=real_embs,
        gen_embs=gen_embs,
        save_path=cfg.paths.detector_path,
        hidden_dims=mc.detector_hidden_dims,
        dropout=mc.detector_dropout,
        lr=mc.detector_lr,
        weight_decay=mc.detector_weight_decay,
        batch_size=mc.detector_batch_size,
        epochs=mc.detector_epochs,
        patience=mc.detector_patience,
        seed=cfg.optimization.seed,
    )

    detector_metrics_path = cfg.paths.results_dir / "detector_metrics.json"
    dataset_stats_path = cfg.paths.results_dir / "dataset_stats.json"
    confusion_plot_path = cfg.paths.plots_dir / "confusion_matrix.png"
    payload = _save_detector_metrics(result, detector_metrics_path, confusion_plot_path)
    dataset_stats = _save_dataset_stats(result, dataset_stats_path)
    torch.save(result.model.state_dict(), cfg.paths.detector_a_path)
    torch.save(result.model.state_dict(), cfg.paths.detector_b_path)

    logger.info(
        "[train_detector] Balanced sample counts | real=%d fake=%d",
        result.balanced_real_count,
        result.balanced_fake_count,
    )
    logger.info("[train_detector] Validation metrics")
    logger.info("  accuracy=%.4f", payload["accuracy"])
    logger.info("  precision=%.4f", payload["precision"])
    logger.info("  recall=%.4f", payload["recall"])
    logger.info("  f1_score=%.4f", payload["f1_score"])
    logger.info("  roc_auc=%.4f", payload["roc_auc"])
    logger.info("  confusion_matrix=%s", payload["confusion_matrix"])
    logger.info("[train_detector] Dataset statistics")
    logger.info(
        "  total_real=%d total_fake=%d train=%d eval=%d class_balance_ratio=%.4f",
        dataset_stats["total_real_samples"],
        dataset_stats["total_fake_samples"],
        dataset_stats["train_count"],
        dataset_stats["eval_count"],
        dataset_stats["class_balance_ratio"],
    )
    logger.info("[train_detector] Metrics JSON -> %s", detector_metrics_path)
    logger.info("[train_detector] Dataset stats JSON -> %s", dataset_stats_path)
    logger.info("[train_detector] Confusion matrix plot -> %s", confusion_plot_path)
    logger.info(
        "[train_detector] Done. Checkpoints -> %s | %s | %s",
        cfg.paths.detector_path,
        cfg.paths.detector_a_path,
        cfg.paths.detector_b_path,
    )


if __name__ == "__main__":
    main()
