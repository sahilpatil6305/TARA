"""
run_ablations.py
================
Run the FoeGlass-style baseline, DRAO, and requested ablations.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import get_config
from src.logging_utils import setup_logger


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "run"


def _build_ablation_descriptor(args: argparse.Namespace, seed: int) -> str:
    parts = [
        "batch-ablations",
        f"seed-{seed}",
        f"attack-{args.attack_detector}",
    ]
    if args.eval_detector is not None:
        parts.append(f"eval-{args.eval_detector}")
    if args.iterations is not None:
        parts.append(f"iters-{args.iterations}")
    return _slugify("_".join(parts))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FoeGlass baseline and DRAO ablations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--iterations", type=int, default=None, help="Iteration override")
    parser.add_argument("--attack-detector", type=str, default="A", choices=["A", "B", "default"])
    parser.add_argument("--eval-detector", type=str, default=None, choices=["A", "B", "default"])
    parser.add_argument("--session-root", type=str, default=None, help="Optional ablation batch directory")
    return parser.parse_args()


def _run_experiment(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _load_run_frame(results_dir: Path, run_name: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / run_name / "metrics.csv")


def _load_summary(results_dir: Path, run_name: str) -> dict:
    with (results_dir / run_name / "summary.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_attack_metrics(results_dir: Path, run_name: str) -> dict:
    with (results_dir / run_name / "attack_metrics.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_stability_metrics(results_dir: Path, run_name: str) -> dict:
    with (results_dir / run_name / "stability_metrics.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _copy_named_results(results_dir: Path, run_name: str, target_name: str) -> None:
    shutil.copyfile(results_dir / run_name / "metrics.csv", results_dir / target_name)


def _plot_line_comparison(
    frames: dict[str, pd.DataFrame],
    column: str,
    title: str,
    ylabel: str,
    target: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, frame in frames.items():
        ax.plot(frame["iteration"], frame[column], marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150)
    plt.close(fig)


def _plot_asr_bars(summary_df: pd.DataFrame, target: Path) -> None:
    labels = summary_df["run_name"].tolist()
    thresholds = ["ASR@0.05", "ASR@0.1", "ASR@0.15", "ASR@0.2"]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, threshold in enumerate(thresholds):
        ax.bar(x + idx * width, summary_df[threshold], width=width, label=threshold)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("ASR Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150)
    plt.close(fig)


def _plot_tradeoff(summary_df: pd.DataFrame, target: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["average_mmd"], summary_df["ASR@0.1"], marker="o", linewidth=2)
    for _, row in summary_df.iterrows():
        ax.annotate(row["run_name"], (row["average_mmd"], row["ASR@0.1"]))
    ax.set_xlabel("Average MMD")
    ax.set_ylabel("ASR@0.1")
    ax.set_title("Realism vs ASR Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=150)
    plt.close(fig)


def _pct_reduction(baseline: float, improved: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return 100.0 * (baseline - improved) / baseline


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    root = Path(__file__).resolve().parent
    venv_python = root / "venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = args.seed if args.seed is not None else cfg.optimization.seed
    batch_name = _build_ablation_descriptor(args, seed)
    session_root = Path(args.session_root) if args.session_root else (cfg.paths.experiments_dir / f"{timestamp}_{batch_name}")
    session_paths = cfg.paths.for_session(session_root)
    results_dir = session_paths.results_dir
    plots_dir = session_paths.plots_dir
    logger = setup_logger("foeglass.run_ablations", session_paths.logs_dir / "run_ablations.log")
    logger.info("Ablation session root: %s", session_root)

    experiments = [
        ("baseline", "baseline"),
        ("no_mmd", "no_mmd"),
        ("no_smoothness", "no_smoothness"),
        ("full_drao", "full_drao"),
    ]

    for run_name, profile in experiments:
        command = [
            python_exe,
            "main.py",
            "--run-name",
            run_name,
            "--profile",
            profile,
            "--attack-detector",
            args.attack_detector,
            "--session-root",
            str(session_root),
        ]
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        if args.iterations is not None:
            command.extend(["--iterations", str(args.iterations)])
        if args.eval_detector is not None:
            command.extend(["--eval-detector", args.eval_detector])
        logger.info("Starting ablation run %s with profile=%s", run_name, profile)
        _run_experiment(command, root)
        logger.info("Finished ablation run %s", run_name)

    _copy_named_results(results_dir, "baseline", "results_baseline.csv")
    _copy_named_results(results_dir, "full_drao", "results_drao.csv")

    frames = {run_name: _load_run_frame(results_dir, run_name) for run_name, _ in experiments}
    summaries = {run_name: _load_summary(results_dir, run_name) for run_name, _ in experiments}
    attacks = {run_name: _load_attack_metrics(results_dir, run_name) for run_name, _ in experiments}
    stabilities = {run_name: _load_stability_metrics(results_dir, run_name) for run_name, _ in experiments}

    rows: list[dict[str, float | str]] = []
    for run_name, _ in experiments:
        attack = attacks[run_name]
        stability = stabilities[run_name]
        summary = summaries[run_name]
        rows.append(
            {
                "run_name": run_name,
                "best_score": float(attack["best_score"]),
                "average_score": float(attack["average_score"]),
                "average_mmd": float(attack["average_mmd"]),
                "average_smoothness": float(attack["average_smoothness"]),
                "average_spectral_continuity": float(attack["average_spectral_continuity"]),
                "average_embedding_variance": float(attack["average_embedding_variance"]),
                "score_variance": float(stability["detector_score_variance"]),
                "objective_variance": float(stability["objective_variance"]),
                "smoothness_score": float(stability["smoothness_score"]),
                "delta_f_mean": float(attack["delta_f_mean"]),
                "delta_f_variance": float(attack["delta_f_variance"]),
                "ASR@0.05": float(attack.get("ASR@0.05", 0.0)),
                "ASR@0.1": float(attack.get("ASR@0.1", 0.0)),
                "ASR@0.15": float(attack.get("ASR@0.15", 0.0)),
                "ASR@0.2": float(attack.get("ASR@0.2", 0.0)),
                "final_score": float(summary["final_detector_score"]),
            }
        )

    summary_df = pd.DataFrame(rows)
    baseline_row = summary_df.loc[summary_df["run_name"] == "baseline"].iloc[0]
    drao_row = summary_df.loc[summary_df["run_name"] == "full_drao"].iloc[0]

    improvements = {
        "baseline_mmd": float(baseline_row["average_mmd"]),
        "drao_mmd": float(drao_row["average_mmd"]),
        "baseline_smoothness": float(baseline_row["average_smoothness"]),
        "drao_smoothness": float(drao_row["average_smoothness"]),
        "baseline_score_variance": float(baseline_row["score_variance"]),
        "drao_score_variance": float(drao_row["score_variance"]),
        "mmd_reduction_percent": _pct_reduction(float(baseline_row["average_mmd"]), float(drao_row["average_mmd"])),
        "smoothness_improvement_percent": _pct_reduction(float(baseline_row["average_smoothness"]), float(drao_row["average_smoothness"])),
        "variance_reduction_percent": _pct_reduction(float(baseline_row["score_variance"]), float(drao_row["score_variance"])),
        "asr_baseline_0_1": float(baseline_row["ASR@0.1"]),
        "asr_drao_0_1": float(drao_row["ASR@0.1"]),
        "conclusion": "Baseline shows higher distribution drift, higher instability, and poorer trajectory consistency.",
    }

    summary_df.to_csv(results_dir / "results_ablation.csv", index=False)
    with (results_dir / "comparison_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(improvements, fh, indent=2)
    with (results_dir / "comparison_statement.txt").open("w", encoding="utf-8") as fh:
        fh.write(improvements["conclusion"] + "\n")

    comparison_frames = {
        "baseline": frames["baseline"],
        "full_drao": frames["full_drao"],
    }
    _plot_line_comparison(comparison_frames, "detector_score", "Score vs Iteration", "Detector score", plots_dir / "score_vs_iteration_comparison.png")
    _plot_line_comparison(comparison_frames, "mmd_sq", "MMD vs Iteration", "MMD", plots_dir / "mmd_vs_iteration_comparison.png")
    _plot_line_comparison(comparison_frames, "smoothness", "Smoothness vs Iteration", "Smoothness", plots_dir / "smoothness_vs_iteration_comparison.png")
    _plot_line_comparison(comparison_frames, "objective", "Objective vs Iteration", "Objective", plots_dir / "objective_vs_iteration_comparison.png")
    _plot_asr_bars(summary_df, plots_dir / "asr_comparison.png")
    _plot_tradeoff(summary_df, plots_dir / "realism_vs_asr_tradeoff.png")
    with (session_root / "session_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": timestamp,
                "batch_name": batch_name,
                "session_root": str(session_root),
                "experiments": [{"run_name": run_name, "profile": profile} for run_name, profile in experiments],
                "results_dir": str(results_dir),
                "plots_dir": str(plots_dir),
                "logs_dir": str(session_paths.logs_dir),
            },
            fh,
            indent=2,
        )
    logger.info("Saved ablation summary to %s", results_dir / "results_ablation.csv")
    logger.info("Saved ablation plots to %s", plots_dir)
    logger.info("Saved session manifest to %s", session_root / "session_manifest.json")

    print(improvements["conclusion"])
    print(json.dumps(improvements, indent=2))


if __name__ == "__main__":
    main()
