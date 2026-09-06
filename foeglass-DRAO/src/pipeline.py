"""
src/pipeline.py
===============
Shared optimisation and evaluation pipeline for the FoeGlass baseline and DRAO.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from . import detector as det_module
from .embedding import Wav2VecEmbedder
from .llm_generator import PromptGenerator, PromptMemory, PromptRecord
from .mmd import EmbeddingBuffer, batch_mmd
from .smoothness import multiscale_temporal_smoothness
from .tts_engine import TTSEngine


LOGGER = logging.getLogger(__name__)


@dataclass
class IterationResult:
    iteration: int
    prompt_mode: str
    reused_prompt: bool
    exploitation_triggered: bool
    mutated_prompt: bool
    elite_source_iteration: int
    transcript: str
    audio_path: str
    detector_score: float
    detector_logit: float
    transfer_detector_score: float
    transfer_detector_logit: float
    detector_signal: float
    detector_signal_norm: float
    mmd_sq: float
    mmd_sq_norm: float
    mmd_bandwidth: float
    smoothness: float
    smoothness_norm: float
    spectral_continuity: float
    embedding_variance: float
    raw_objective: float
    momentum_objective: float
    objective: float
    delta_f: float
    absolute_f: float
    margin_term: float
    delta_d: float
    wall_time_s: float


@dataclass
class RunningZScore:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> float:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        return self.normalize(value)

    def normalize(self, value: float) -> float:
        if self.count < 2:
            return 0.0
        variance = self.m2 / max(self.count - 1, 1)
        std = float(np.sqrt(max(variance, 1e-12)))
        return float((value - self.mean) / std)


def _relative_delta(current: float, previous: Optional[float]) -> float:
    if previous is None:
        return 0.0
    scale = max(abs(previous), 1.0)
    return float((current - previous) / (scale + 1e-6))


def _to_logit(prob: float) -> float:
    clipped = min(max(prob, 1e-6), 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped + 1e-6)))


def _safe_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.var(values))


def _longest_improving_streak(values: Sequence[float]) -> int:
    if not values:
        return 0
    best = 1
    current = 1
    for prev, curr in zip(values, values[1:]):
        if curr > prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _spectral_continuity(waveform: np.ndarray, sample_rate: int) -> float:
    del sample_rate
    stft = np.abs(librosa.stft(waveform, n_fft=512, hop_length=128))
    if stft.shape[1] < 2:
        return 1.0
    a = stft[:, :-1]
    b = stft[:, 1:]
    numer = np.sum(a * b, axis=0)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    cosine = numer / np.clip(denom, 1e-8, None)
    return float(np.mean(cosine))


def _load_waveform(audio_path: Path, sample_rate: int) -> np.ndarray:
    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != sample_rate:
        waveform = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=sample_rate)
    return np.asarray(waveform, dtype=np.float32)


def _compute_audio_quality(
    audio_paths: Sequence[Path],
    sample_rate: int,
    clean_reference_paths: Optional[Sequence[Path]] = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "reference_metrics_used": False,
        "pesq": None,
        "stoi": None,
    }

    continuity_scores: list[float] = []
    loudness_scores: list[float] = []
    for path in audio_paths:
        waveform = _load_waveform(path, sample_rate)
        continuity_scores.append(_spectral_continuity(waveform, sample_rate))
        rms = librosa.feature.rms(y=waveform, frame_length=512, hop_length=128)[0]
        loudness_scores.append(float(np.var(rms)) if rms.size else 0.0)

    metrics["spectral_continuity"] = float(np.mean(continuity_scores)) if continuity_scores else 0.0
    metrics["loudness_variance"] = float(np.mean(loudness_scores)) if loudness_scores else 0.0

    if clean_reference_paths and len(clean_reference_paths) == len(audio_paths):
        try:
            from pesq import pesq  # type: ignore
            from pystoi import stoi  # type: ignore

            pesq_scores: list[float] = []
            stoi_scores: list[float] = []
            for clean_path, degraded_path in zip(clean_reference_paths, audio_paths):
                clean_wave = _load_waveform(clean_path, sample_rate)
                deg_wave = _load_waveform(degraded_path, sample_rate)
                min_len = min(len(clean_wave), len(deg_wave))
                if min_len < 256:
                    continue
                clean_wave = clean_wave[:min_len]
                deg_wave = deg_wave[:min_len]
                pesq_scores.append(float(pesq(sample_rate, clean_wave, deg_wave, "wb")))
                stoi_scores.append(float(stoi(clean_wave, deg_wave, sample_rate, extended=False)))
            metrics["reference_metrics_used"] = bool(pesq_scores or stoi_scores)
            metrics["pesq"] = float(np.mean(pesq_scores)) if pesq_scores else None
            metrics["stoi"] = float(np.mean(stoi_scores)) if stoi_scores else None
        except Exception:
            metrics["reference_metrics_used"] = False

    return metrics


def _format_threshold(threshold: float) -> str:
    return f"{threshold:.2f}".rstrip("0").rstrip(".")


def _build_asr_metrics(
    scores: Sequence[float],
    thresholds: Sequence[float],
    transfer_scores: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        f"ASR@{_format_threshold(threshold)}": float(np.mean(np.asarray(scores) >= threshold))
        for threshold in thresholds
    }
    payload["max_detector_score"] = float(np.max(scores)) if scores else 0.0
    payload["average_detector_score"] = float(np.mean(scores)) if scores else 0.0
    payload["detector_score_variance"] = _safe_variance(scores)
    if transfer_scores is not None:
        payload["transfer_success_rate"] = float(
            np.mean(np.asarray(transfer_scores) >= 0.5)
        ) if transfer_scores else 0.0
    return payload


def _compute_stability_metrics(results: Sequence[IterationResult]) -> dict[str, Any]:
    raw_objectives = [r.raw_objective for r in results]
    scores = [r.detector_score for r in results]
    smoothness_scores = [r.smoothness for r in results]
    delta_f_values = [r.delta_f for r in results]
    return {
        "objective_variance": _safe_variance(raw_objectives),
        "detector_score_variance": _safe_variance(scores),
        "smoothness_score": float(np.mean(smoothness_scores)) if smoothness_scores else 0.0,
        "positive_objective_iterations": int(sum(1 for value in raw_objectives if value > 0.0)),
        "positive_delta_f_percent": float(
            100.0 * np.mean(np.asarray(delta_f_values) > 0.0)
        ) if delta_f_values else 0.0,
        "longest_improving_streak": _longest_improving_streak(raw_objectives),
    }


def _save_json(payload: Mapping[str, Any], *targets: Path) -> None:
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def _plot_series(
    iters: Sequence[int],
    values: Sequence[float],
    title: str,
    ylabel: str,
    colour: str,
    targets: Sequence[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, values, marker="o", linewidth=2, color=colour)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, dpi=150)
    plt.close(fig)


def _plot_asr_summary(asr_metrics: Mapping[str, Any], targets: Sequence[Path]) -> None:
    labels = [key for key in asr_metrics if key.startswith("ASR@")]
    values = [float(asr_metrics[key]) for key in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4063d8", "#389826", "#cb3c33", "#9558b2"][: len(labels)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("ASR Summary")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, dpi=150)
    plt.close(fig)


def _plot_detector_histogram(scores: Sequence[float], targets: Sequence[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=min(10, max(len(scores), 1)), color="tab:blue", alpha=0.8, edgecolor="black")
    ax.set_title("Detector Score Distribution")
    ax.set_xlabel("Detector score")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, dpi=150)
    plt.close(fig)


def _plot_metrics(
    results: List[IterationResult],
    run_dir: Path,
    plots_dir: Path,
    run_name: str,
) -> None:
    iters = [r.iteration for r in results]
    specs = [
        ("score_vs_iteration.png", [r.detector_score for r in results], "Detector score", "Score", "tab:blue"),
        ("logit_vs_iteration.png", [r.detector_logit for r in results], "Detector logit", "Logit", "tab:purple"),
        ("mmd_vs_iteration.png", [r.mmd_sq for r in results], "MMD", "MMD^2", "tab:orange"),
        ("smoothness_vs_iteration.png", [r.smoothness for r in results], "Smoothness", "Smoothness", "tab:green"),
        ("objective_vs_iteration.png", [r.objective for r in results], "Objective", "J_t", "tab:red"),
    ]
    for filename, values, title, ylabel, colour in specs:
        _plot_series(
            iters,
            values,
            title,
            ylabel,
            colour,
            [run_dir / filename, plots_dir / f"{run_name}_{filename}"],
        )


def _choose_prompt(
    llm: PromptGenerator,
    memory: PromptMemory,
    iteration: int,
    rng: np.random.Generator,
    use_memory: bool,
    reuse_probability: float,
    best_prompt: Optional[str],
    best_score: float,
    exploitation_threshold: float,
    exploitation_probability: float,
) -> tuple[str, str, bool, bool, Optional[PromptRecord]]:
    exploitation_triggered = (
        use_memory and
        best_prompt is not None and
        best_score > exploitation_threshold and
        bool(rng.random() < exploitation_probability)
    )
    if exploitation_triggered and best_prompt is not None:
        return (
            llm.mutate(best_prompt, memory, iteration),
            "exploit",
            True,
            True,
            None,
        )

    top_record = memory.top_prompt()
    reuse_elite = (
        use_memory and
        top_record is not None and
        bool(rng.random() < reuse_probability)
    )
    if reuse_elite and top_record is not None:
        return (
            llm.mutate(top_record.transcript, memory, iteration),
            "reused",
            True,
            False,
            top_record,
        )
    return llm.generate(memory, iteration), "new", False, False, None


def _resolve_mode_settings(
    mode: str,
    use_mmd: bool,
    use_smoothness: bool,
    use_momentum: bool,
    use_delta_f: bool,
    use_memory: bool,
    normalize_terms: bool,
) -> dict[str, Any]:
    mode = mode.strip().lower()
    settings = {
        "mode": mode,
        "use_mmd": use_mmd,
        "use_smoothness": use_smoothness,
        "use_momentum": use_momentum,
        "use_delta_f": use_delta_f,
        "use_memory": use_memory,
        "normalize_terms": normalize_terms,
    }
    if mode == "baseline":
        settings.update({
            "use_mmd": False,
            "use_smoothness": False,
            "use_momentum": False,
            "use_delta_f": False,
            "normalize_terms": False,
        })
    elif mode == "no_mmd":
        settings["use_mmd"] = False
    elif mode == "no_smoothness":
        settings["use_smoothness"] = False
    elif mode == "no_momentum":
        settings["use_momentum"] = False
    return settings


def _read_optimization_config(config_snapshot: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not config_snapshot:
        return {}
    optimization = config_snapshot.get("optimization", {})
    return optimization if isinstance(optimization, Mapping) else {}


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def run_pipeline(
    embedder: Wav2VecEmbedder,
    detector: det_module.DetectorMLP,
    llm: PromptGenerator,
    tts: TTSEngine,
    real_embeddings: np.ndarray,
    audio_dir: Path,
    results_dir: Path,
    plots_dir: Path,
    logs_dir: Path,
    iterations: int = 20,
    alpha_attack_gain: float = 3.0,
    beta_absolute_score: float = 1.0,
    margin_weight: float = 0.25,
    lambda_mmd: float = 0.2,
    lambda_smoothness: float = 0.1,
    buffer_size: int = 20,
    memory_top_k: int = 5,
    memory_reuse_probability: float = 0.5,
    momentum_gamma: float = 0.6,
    asr_thresholds: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
    margin_target: float = 0.2,
    initial_mmd: float = 1_000_000.0,
    run_name: str = "run_0",
    real_subset_size: int = 128,
    seed: int = 42,
    use_mmd: bool = True,
    use_smoothness: bool = True,
    use_logit: bool = False,
    use_memory: bool = True,
    use_margin: bool = True,
    use_delta_score: bool = True,
    use_momentum: bool = True,
    normalize_terms: bool = True,
    memory_metric: str = "objective",
    profile_name: str = "full_drao",
    transfer_detector: Optional[det_module.DetectorMLP] = None,
    clean_reference_paths: Optional[Sequence[Path]] = None,
    config_snapshot: Optional[Mapping[str, Any]] = None,
) -> List[IterationResult]:
    del initial_mmd

    optimization_cfg = _read_optimization_config(config_snapshot)
    mode_name = str(optimization_cfg.get("mode", profile_name or "drao"))
    resolved = _resolve_mode_settings(
        mode=mode_name,
        use_mmd=bool(optimization_cfg.get("use_mmd", use_mmd)),
        use_smoothness=bool(optimization_cfg.get("use_smoothness", use_smoothness)),
        use_momentum=bool(optimization_cfg.get("use_momentum", use_momentum)),
        use_delta_f=bool(optimization_cfg.get("use_delta_f", use_delta_score)),
        use_memory=bool(optimization_cfg.get("use_memory", use_memory)),
        normalize_terms=bool(optimization_cfg.get("normalize_terms", normalize_terms)),
    )
    use_mmd = bool(resolved["use_mmd"])
    use_smoothness = bool(resolved["use_smoothness"])
    use_momentum = bool(resolved["use_momentum"])
    use_delta_f = bool(resolved["use_delta_f"])
    use_memory = bool(resolved["use_memory"])
    normalize_terms = bool(resolved["normalize_terms"])
    profile_name = str(resolved["mode"])

    exploitation_threshold = float(optimization_cfg.get("exploitation_threshold", 0.05))
    exploitation_probability = float(optimization_cfg.get("exploitation_probability", memory_reuse_probability))
    adaptive_score_threshold = float(optimization_cfg.get("adaptive_score_threshold", 0.1))
    adaptive_decay_factor = float(optimization_cfg.get("adaptive_decay_factor", 0.5))
    original_lambda_mmd = lambda_mmd
    original_lambda_smoothness = lambda_smoothness

    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    memory = PromptMemory(top_k=memory_top_k)
    emb_buf = EmbeddingBuffer(max_size=buffer_size)
    history: List[IterationResult] = []

    prev_signal_value: Optional[float] = None
    prev_d_value: Optional[float] = None
    prev_momentum_objective = 0.0
    best_prompt: Optional[str] = None
    best_score = float("-inf")
    signal_norm = RunningZScore()
    d_norm = RunningZScore()
    s_norm = RunningZScore()

    rng = np.random.default_rng(seed)
    n_real = len(real_embeddings)
    subset_idx = rng.choice(n_real, min(real_subset_size, n_real), replace=False)
    real_subset = real_embeddings[subset_idx]

    logger = logging.getLogger("foeglass.main")
    logger.info(
        f"\n{'=' * 60}\n"
        f"  Experiment Pipeline  -  {run_name}\n"
        f"  profile={profile_name} iterations={iterations} alpha={alpha_attack_gain} beta={beta_absolute_score} "
        f"margin={margin_weight} lambda1={lambda_mmd} lambda2={lambda_smoothness} "
        f"use_mmd={use_mmd} use_smoothness={use_smoothness} use_delta={use_delta_f} "
        f"use_momentum={use_momentum} normalize_terms={normalize_terms} memory_metric={memory_metric}\n"
        f"{'=' * 60}"
    )

    csv_path = run_dir / "metrics.csv"
    full_log_path = logs_dir / f"{run_name}_full_log.csv"
    csv_fields = [f.name for f in IterationResult.__dataclass_fields__.values()]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_fh, full_log_path.open(
        "w", newline="", encoding="utf-8"
    ) as full_log_fh:
        run_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        full_writer = csv.DictWriter(full_log_fh, fieldnames=csv_fields)
        run_writer.writeheader()
        full_writer.writeheader()

        for t in range(1, iterations + 1):
            t_start = time.perf_counter()
            transcript, prompt_mode, reused_prompt, exploitation_triggered, elite_record = _choose_prompt(
                llm=llm,
                memory=memory,
                iteration=t,
                rng=rng,
                use_memory=use_memory,
                reuse_probability=memory_reuse_probability,
                best_prompt=best_prompt,
                best_score=best_score,
                exploitation_threshold=exploitation_threshold,
                exploitation_probability=exploitation_probability,
            )
            mutated = prompt_mode in {"reused", "exploit"}
            elite_source_iteration = elite_record.iteration if elite_record else -1

            logger.info("\n[iter %3d/%d]", t, iterations)
            logger.info(
                "  prompt_mode=%s mutated=%s reused=%s exploited=%s source_iter=%s",
                prompt_mode,
                mutated,
                reused_prompt,
                exploitation_triggered,
                elite_source_iteration,
            )
            logger.info("  prompt=%r%s", transcript[:180], "..." if len(transcript) > 180 else "")

            audio_path = tts.synthesise(
                transcript,
                output_dir=audio_dir / run_name,
                stem=f"t{t:03d}",
            )

            emb_out = embedder.extract(audio_path)
            g_emb = emb_out.global_embedding
            f_emb = emb_out.frame_embeddings
            waveform = _load_waveform(audio_path, tts.sample_rate)

            detector_score = det_module.score(detector, g_emb)
            detector_logit = _to_logit(detector_score)
            transfer_score = det_module.score(transfer_detector, g_emb) if transfer_detector is not None else detector_score
            transfer_logit = _to_logit(transfer_score)
            emb_buf.append(g_emb.cpu().numpy())

            if emb_buf.is_ready(min_size=2):
                mmd_sq, bandwidth = batch_mmd(emb_buf.as_array(), real_subset)
                embedding_variance = float(np.mean(np.var(emb_buf.as_array(), axis=0)))
            else:
                mmd_sq, bandwidth = 0.0, 0.0
                embedding_variance = 0.0

            smoothness = float(multiscale_temporal_smoothness(f_emb))
            spectral_continuity = _spectral_continuity(waveform, tts.sample_rate)
            detector_signal = detector_logit if use_logit else detector_score

            if profile_name == "adaptive_constraints":
                if detector_score > adaptive_score_threshold:
                    lambda_mmd = original_lambda_mmd * adaptive_decay_factor
                    lambda_smoothness = original_lambda_smoothness * adaptive_decay_factor
                else:
                    lambda_mmd = original_lambda_mmd
                    lambda_smoothness = original_lambda_smoothness
            else:
                lambda_mmd = original_lambda_mmd
                lambda_smoothness = original_lambda_smoothness

            detector_signal_norm = signal_norm.update(detector_signal) if normalize_terms else detector_signal
            mmd_sq_norm = d_norm.update(mmd_sq) if normalize_terms else mmd_sq
            smoothness_norm = s_norm.update(smoothness) if normalize_terms else smoothness

            signal_value = detector_signal_norm if normalize_terms else detector_signal
            d_value = mmd_sq_norm if normalize_terms else mmd_sq
            smoothness_value = smoothness_norm if normalize_terms else smoothness

            delta_f = (signal_value - prev_signal_value) if (use_delta_f and prev_signal_value is not None) else 0.0
            absolute_f = detector_score if profile_name == "baseline" else signal_value
            margin_term = max(0.0, margin_target - detector_score) if use_margin else 0.0
            delta_d = (d_value - prev_d_value) if (use_mmd and prev_d_value is not None) else 0.0

            if profile_name == "baseline":
                raw_objective = detector_score
            else:
                raw_objective = (
                    alpha_attack_gain * delta_f
                    + beta_absolute_score * absolute_f
                    + margin_weight * margin_term
                    - (lambda_mmd * delta_d if use_mmd else 0.0)
                    - (lambda_smoothness * smoothness_value if use_smoothness else 0.0)
                )
            momentum_objective = (
                momentum_gamma * prev_momentum_objective + raw_objective
                if use_momentum else raw_objective
            )
            selection_objective = detector_score if memory_metric == "detector_score" else momentum_objective
            wall_t = time.perf_counter() - t_start

            result = IterationResult(
                iteration=t,
                prompt_mode=prompt_mode,
                reused_prompt=reused_prompt,
                exploitation_triggered=exploitation_triggered,
                mutated_prompt=mutated,
                elite_source_iteration=elite_source_iteration,
                transcript=transcript,
                audio_path=str(audio_path),
                detector_score=round(detector_score, 6),
                detector_logit=round(detector_logit, 6),
                transfer_detector_score=round(transfer_score, 6),
                transfer_detector_logit=round(transfer_logit, 6),
                detector_signal=round(detector_signal, 6),
                detector_signal_norm=round(detector_signal_norm, 6),
                mmd_sq=round(mmd_sq, 6),
                mmd_sq_norm=round(mmd_sq_norm, 6),
                mmd_bandwidth=round(bandwidth, 6),
                smoothness=round(smoothness, 6),
                smoothness_norm=round(smoothness_norm, 6),
                spectral_continuity=round(spectral_continuity, 6),
                embedding_variance=round(embedding_variance, 6),
                raw_objective=round(raw_objective, 6),
                momentum_objective=round(momentum_objective, 6),
                objective=round(momentum_objective if use_momentum else raw_objective, 6),
                delta_f=round(delta_f, 6),
                absolute_f=round(absolute_f, 6),
                margin_term=round(margin_term, 6),
                delta_d=round(delta_d, 6),
                wall_time_s=round(wall_t, 3),
            )
            history.append(result)
            row = asdict(result)
            run_writer.writerow(row)
            full_writer.writerow(row)
            csv_fh.flush()
            full_log_fh.flush()

            logger.info(
                "  score=%.4f logit=%+.4f transfer_score=%.4f mmd=%.4f smoothness=%.4f spectral=%.4f J=%+.4f",
                detector_score,
                detector_logit,
                transfer_score,
                mmd_sq,
                smoothness,
                spectral_continuity,
                result.objective,
            )

            memory.add(
                transcript=transcript,
                detector_score=detector_score,
                objective=selection_objective,
                iteration=t,
            )
            if detector_score > best_score:
                best_score = detector_score
                best_prompt = transcript
            prev_signal_value = signal_value
            prev_d_value = d_value
            prev_momentum_objective = momentum_objective

    best = max(history, key=lambda r: r.objective)
    first = history[0]
    k = min(5, len(history))
    first_window = history[:k]
    last_window = history[-k:]
    detector_scores = [r.detector_score for r in history]
    detector_logits = [r.detector_logit for r in history]
    mmd_values = [r.mmd_sq for r in history]
    smoothness_values = [r.smoothness for r in history]
    spectral_values = [r.spectral_continuity for r in history]
    embedding_variances = [r.embedding_variance for r in history]
    raw_objectives = [r.raw_objective for r in history]
    objectives = [r.objective for r in history]
    delta_f_values = [r.delta_f for r in history]
    transfer_scores = [r.transfer_detector_score for r in history] if transfer_detector is not None else None

    attack_metrics: dict[str, Any] = {
        "run_name": run_name,
        "profile": profile_name,
        "iterations": iterations,
        "best_score": float(np.max(detector_scores)),
        "average_score": float(np.mean(detector_scores)),
        "final_score": float(history[-1].detector_score),
        "success_drop": float(np.max(detector_scores) - history[-1].detector_score),
        "best_mmd": float(np.min(mmd_values)),
        "average_mmd": float(np.mean(mmd_values)),
        "average_smoothness": float(np.mean(smoothness_values)),
        "average_spectral_continuity": float(np.mean(spectral_values)),
        "average_embedding_variance": float(np.mean(embedding_variances)),
        "objective_stability": _safe_variance(raw_objectives),
        "delta_f_mean": float(np.mean(delta_f_values)),
        "delta_f_variance": _safe_variance(delta_f_values),
        "transfer_evaluated": transfer_detector is not None,
    }
    attack_metrics.update(_build_asr_metrics(detector_scores, asr_thresholds, transfer_scores))
    success_frequency = float(attack_metrics.get("ASR@0.1", 0.0))
    attack_metrics["success_frequency"] = success_frequency

    stability_metrics = _compute_stability_metrics(history)
    audio_quality = _compute_audio_quality(
        [Path(r.audio_path) for r in history],
        sample_rate=tts.sample_rate,
        clean_reference_paths=clean_reference_paths,
    )

    validation = {
        "passed": (
            best.objective >= first.objective and
            max(detector_scores) >= first.detector_score and
            float(np.mean([r.objective for r in last_window])) >=
            float(np.mean([r.objective for r in first_window]))
        ),
        "criteria": {
            "best_objective_exceeds_iter1": best.objective >= first.objective,
            "best_detector_score_exceeds_iter1": max(detector_scores) >= first.detector_score,
            "last_window_mean_objective_exceeds_first_window":
                float(np.mean([r.objective for r in last_window])) >=
                float(np.mean([r.objective for r in first_window])),
        },
    }

    elite_payload = [
        {
            "prompt": record.transcript,
            "detector_score": record.detector_score,
            "objective": record.objective,
            "iteration": record.iteration,
        }
        for record in memory.records()
    ]

    run_metadata = {
        "run_name": run_name,
        "profile": profile_name,
        "timestamp": datetime.now().isoformat(),
        "random_seed": seed,
        "subset_indices": subset_idx.tolist(),
        "config": _sanitize_json_value(dict(config_snapshot or {})),
    }

    summary = {
        "run_name": run_name,
        "profile": profile_name,
        "iterations": iterations,
        "alpha_attack_gain": alpha_attack_gain,
        "beta_absolute_score": beta_absolute_score,
        "margin_weight": margin_weight,
        "margin_target": margin_target,
        "lambda_mmd": lambda_mmd,
        "lambda_smoothness": lambda_smoothness,
        "momentum_gamma": momentum_gamma,
        "seed": seed,
        "real_subset_size": len(subset_idx),
        "real_subset_indices": subset_idx.tolist(),
        "use_mmd": use_mmd,
        "use_smoothness": use_smoothness,
        "use_logit": use_logit,
        "use_memory": use_memory,
        "use_margin": use_margin,
        "use_delta_f": use_delta_f,
        "use_momentum": use_momentum,
        "normalize_terms": normalize_terms,
        "memory_metric": memory_metric,
        "best_objective": best.objective,
        "best_iteration": best.iteration,
        "best_transcript": best.transcript,
        "best_detector_score": best.detector_score,
        "final_detector_score": history[-1].detector_score,
        "success_drop": float(np.max(detector_scores) - history[-1].detector_score),
        "success_frequency": success_frequency,
        "best_prompt": best_prompt,
        "exploitation_threshold": exploitation_threshold,
        "exploitation_probability": exploitation_probability,
        "adaptive_score_threshold": adaptive_score_threshold,
        "adaptive_decay_factor": adaptive_decay_factor,
        "detector_score_progression": detector_scores,
        "detector_logit_progression": detector_logits,
        "mmd_progression": mmd_values,
        "smoothness_progression": smoothness_values,
        "spectral_continuity_progression": spectral_values,
        "embedding_variance_progression": embedding_variances,
        "objective_progression": objectives,
        "raw_objective_progression": raw_objectives,
        "delta_f_progression": delta_f_values,
        "elite_prompts": elite_payload,
        "validation_passed": validation["passed"],
    }

    _save_json(summary, run_dir / "summary.json")
    _save_json(validation, run_dir / "validation.json")
    _save_json(attack_metrics, run_dir / "attack_metrics.json", results_dir / "attack_metrics.json")
    _save_json(stability_metrics, run_dir / "stability_metrics.json", results_dir / "stability_metrics.json")
    _save_json(audio_quality, run_dir / "audio_quality.json", results_dir / "audio_quality.json")
    _save_json(run_metadata, run_dir / "run_metadata.json", results_dir / "run_metadata.json")
    _save_json(summary, results_dir / f"summary_{profile_name}.json")
    _save_json(stability_metrics, results_dir / f"stability_{profile_name}.json")
    with (results_dir / f"results_{profile_name}.csv").open("w", newline="", encoding="utf-8") as mode_csv_fh:
        mode_writer = csv.DictWriter(mode_csv_fh, fieldnames=csv_fields)
        mode_writer.writeheader()
        for result in history:
            mode_writer.writerow(asdict(result))

    _plot_metrics(history, run_dir, plots_dir, run_name)
    _plot_asr_summary(
        attack_metrics,
        [run_dir / "asr_summary.png", plots_dir / f"{run_name}_asr_summary.png"],
    )
    _plot_detector_histogram(
        detector_scores,
        [
            run_dir / "detector_distribution_histogram.png",
            plots_dir / f"{run_name}_detector_distribution_histogram.png",
        ],
    )

    logger.info(
        f"\n{'=' * 60}\n"
        f"  Run complete. Best objective = {best.objective:+.4f} at iter {best.iteration}\n"
        f"  ASR@0.05={attack_metrics.get('ASR@0.05', 0.0):.2%} "
        f"ASR@0.1={attack_metrics.get('ASR@0.1', 0.0):.2%} "
        f"ASR@0.15={attack_metrics.get('ASR@0.15', 0.0):.2%} "
        f"ASR@0.2={attack_metrics.get('ASR@0.2', 0.0):.2%}\n"
        f"{'=' * 60}"
    )
    logger.info("Summary JSON: %s", run_dir / "summary.json")
    logger.info("Attack metrics JSON: %s", run_dir / "attack_metrics.json")
    logger.info("Stability metrics JSON: %s", run_dir / "stability_metrics.json")
    logger.info("Audio quality JSON: %s", run_dir / "audio_quality.json")
    logger.info("Run metadata JSON: %s", run_dir / "run_metadata.json")
    logger.info("Metrics CSV: %s", run_dir / "metrics.csv")
    logger.info("Full run CSV: %s", full_log_path)
    return history
