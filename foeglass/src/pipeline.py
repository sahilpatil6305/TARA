import csv
import datetime
import json
import os
import random
import shutil
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from build_real_embeddings import build_real_embeddings
from src.config import DRAO_LAMBDA, REAL_EMBEDDINGS_PATH, REAL_TRAIN_DIR
from src.detector import Detector
from src.distance import compute_distance
from src.diversity import diversity_score
from src.embedding import get_embedding
from src.feedback import generate_feedback_context
from src.memory import Memory
from src.prompt_generator import generate_baseline_prompt, generate_candidates
from src.tts_engine import generate_audio


def save_timestamps(prompt, output_file, audio_path):
    """
    Generate word-level timestamps using the real audio duration.
    """
    words = prompt.split()
    if not words:
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("")
        return

    import librosa

    duration = librosa.get_duration(path=audio_path)
    time_per_word = duration / len(words) if words else 0.0

    with open(output_file, "w", encoding="utf-8") as handle:
        current_time = 0.0
        for word in words:
            end_time = current_time + time_per_word
            handle.write(f"{current_time:.2f} - {end_time:.2f} : {word}\n")
            current_time = end_time


def _write_scores_csv(scores_file, memory):
    history = memory.get_all()
    with open(scores_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iteration",
                "score",
                "diversity_score",
                "distance",
                "objective",
                "final_score",
                "score_delta",
                "mode",
            ]
        )
        for idx, (real_score, div_score, distance_score, objective_score, final_score, score_delta, scoring_mode) in enumerate(
            zip(
                history["real_scores"],
                history["diversity_scores"],
                history["distance_scores"],
                history["objective_scores"],
                history["final_scores"],
                history["score_deltas"],
                history["scoring_modes"],
            ),
            start=1,
        ):
            writer.writerow(
                [
                    idx,
                    real_score,
                    div_score,
                    "" if distance_score is None else distance_score,
                    objective_score,
                    final_score,
                    score_delta,
                    scoring_mode,
                ]
            )


def _save_plot(plot_path, memory):
    history = memory.get_all()
    x = range(1, len(history["final_scores"]) + 1)
    has_distance = any(value is not None for value in history["distance_scores"])

    plt.figure(figsize=(10, 5))
    plt.plot(x, history["real_scores"], label="Detector Score")
    if has_distance:
        plt.plot(x, history["distance_scores"], label="Distance")
        plt.plot(x, history["objective_scores"], label="Objective")
    else:
        plt.plot(x, history["diversity_scores"], label="Diversity Score")
    plt.plot(x, history["final_scores"], label="Final Score")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Adversarial Optimization Loop Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def _extract_context_field(context, field_name, default="NONE"):
    prefix = f"{field_name}:"
    for line in context.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return default


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


@lru_cache(maxsize=4)
def _load_real_embeddings_cached(embeddings_path, modified_time):
    del modified_time
    return np.load(embeddings_path)


def _get_real_embeddings(embeddings_path=REAL_EMBEDDINGS_PATH, real_dir=REAL_TRAIN_DIR):
    resolved_path = os.path.abspath(embeddings_path)
    if not os.path.exists(resolved_path):
        build_real_embeddings(real_dir=real_dir, output_path=resolved_path, force=False)
    return _load_real_embeddings_cached(resolved_path, os.path.getmtime(resolved_path))


def _score_candidate(prompt, audio_path, detector, prompt_history, mode, drao_lambda, real_embeddings=None):
    real_score, feature_map = detector.score_with_features(audio_path)
    result = {
        "prompt": prompt,
        "audio_path": audio_path,
        "real_score": real_score,
        "score": real_score,
        "attack_success": real_score >= 0.5,
        "feature_map": feature_map,
    }

    if mode == "drao":
        if real_embeddings is None:
            raise ValueError("[!] real_embeddings are required for DRAO scoring.")
        embedding = get_embedding(audio_path)
        distance = compute_distance(embedding, real_embeddings)
        objective = real_score - (drao_lambda * distance)
        result.update(
            {
                "embedding": embedding,
                "distance": distance,
                "objective": objective,
                "diversity_score": 0.0,
                "final_score": objective,
            }
        )
        return result

    div_score = diversity_score(prompt, prompt_history)
    final_score = 0.7 * real_score + 0.3 * div_score
    result.update(
        {
            "distance": None,
            "objective": final_score,
            "diversity_score": div_score,
            "final_score": final_score,
        }
    )
    return result


def _candidate_sort_key(candidate):
    distance = candidate["distance"] if candidate["distance"] is not None else float("inf")
    return candidate["final_score"], candidate["real_score"], -distance


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _candidate_prompts_for_iteration(memory, mode, seed, iteration, candidates_per_iteration):
    if mode == "baseline":
        return [generate_baseline_prompt(iteration, seed)]

    feedback_context = generate_feedback_context(memory)
    parent_prompt = _extract_context_field(feedback_context, "BEST_PROMPT", "NONE")
    prompts = generate_candidates(feedback_context, num_candidates=candidates_per_iteration)
    return prompts, parent_prompt


def run_pipeline(iterations=20, candidates_per_iteration=4, mode="optimized", seed=42, drao_lambda=DRAO_LAMBDA):
    _set_seed(seed)
    mode_key = mode.lower()
    if mode_key not in {"baseline", "optimized", "drao"}:
        raise ValueError("[!] mode must be one of: baseline, optimized, DRAO")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = "baseline" if mode_key == "baseline" else "run"
    run_results_dir = os.path.join("results", f"{run_prefix}_{run_id}_seed{seed}")
    run_audio_dir = os.path.join("data", "audio", f"{run_prefix}_{run_id}_seed{seed}")

    os.makedirs(run_results_dir, exist_ok=True)
    os.makedirs(os.path.join(run_results_dir, "plots"), exist_ok=True)
    os.makedirs(run_audio_dir, exist_ok=True)

    print(f"[*] === Starting {mode.title()} Pipeline ({iterations} iterations) ===")
    print(f"[*] Audio will be saved to: {run_audio_dir}")
    print(f"[*] Results will be saved to: {run_results_dir}")
    print(f"[*] Candidates per iteration: {1 if mode_key == 'baseline' else candidates_per_iteration}")
    print(f"[*] Seed: {seed}")
    if mode_key == "drao":
        print(f"[*] DRAO lambda: {drao_lambda:.4f}")

    memory = Memory()
    detector = Detector()
    scores_file = os.path.join(run_results_dir, "scores.csv")
    real_embeddings = _get_real_embeddings() if mode_key == "drao" else None

    for iteration in range(1, iterations + 1):
        print(f"\n[*] --- Iteration {iteration} ({mode}) ---")
        iteration_dir = os.path.join(run_audio_dir, f"iteration_{iteration:03d}")
        candidates_dir = os.path.join(iteration_dir, "candidates")
        os.makedirs(candidates_dir, exist_ok=True)

        parent_prompt = "NONE"
        if mode_key == "baseline":
            candidate_prompts = [generate_baseline_prompt(iteration, seed)]
        else:
            feedback_context = generate_feedback_context(memory)
            parent_prompt = _extract_context_field(feedback_context, "BEST_PROMPT", "NONE")
            candidate_prompts = generate_candidates(feedback_context, num_candidates=candidates_per_iteration)

        print(f"[*] Generated {len(candidate_prompts)} prompt candidates")
        for idx, prompt in enumerate(candidate_prompts, start=1):
            print(f"[*] Candidate {idx}: {prompt}")

        prompt_history = list(memory.get_all()["embeddings"])
        candidate_results = []

        for idx, prompt in enumerate(candidate_prompts, start=1):
            candidate_audio_path = os.path.join(candidates_dir, f"candidate_{idx:02d}.wav")
            generate_audio(prompt, candidate_audio_path)
            result = _score_candidate(
                prompt,
                candidate_audio_path,
                detector,
                prompt_history,
                mode=mode_key,
                drao_lambda=drao_lambda,
                real_embeddings=real_embeddings,
            )
            result["candidate_index"] = idx
            candidate_results.append(result)

        best_candidate = max(
            candidate_results,
            key=_candidate_sort_key,
        )

        selected_audio_path = os.path.join(iteration_dir, "audio.wav")
        shutil.copyfile(best_candidate["audio_path"], selected_audio_path)

        print(
            "[*] Selected candidate "
            f"{best_candidate['candidate_index']} | "
            f"score={best_candidate['score']:.4f} | "
            f"{'distance=' + format(best_candidate['distance'], '.4f') + ' | ' if best_candidate['distance'] is not None else ''}"
            f"objective={best_candidate['objective']:.4f}"
        )

        metadata = {
            "iteration": iteration,
            "prompt": best_candidate["prompt"],
            "mode": mode,
            "score": best_candidate["score"],
            "real_score": best_candidate["real_score"],
            "diversity_score": best_candidate["diversity_score"],
            "distance": best_candidate["distance"],
            "objective": best_candidate["objective"],
            "final_score": best_candidate["final_score"],
            "lambda": drao_lambda if mode_key == "drao" else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "attack_success": best_candidate["attack_success"],
        }
        _save_json(os.path.join(iteration_dir, "metadata.json"), metadata)

        features_payload = {
            "iteration": iteration,
            "real_score": best_candidate["real_score"],
            "feature_names": best_candidate["feature_map"]["feature_names"],
            "feature_values": best_candidate["feature_map"]["feature_values"],
            "timestamp": metadata["timestamp"],
        }
        _save_json(os.path.join(iteration_dir, "features.json"), features_payload)

        candidate_search_payload = {
            "iteration": iteration,
            "mode": mode,
            "selected_candidate_index": best_candidate["candidate_index"],
            "selected_prompt": best_candidate["prompt"],
            "parent_prompt": None if parent_prompt == "NONE" else parent_prompt,
            "candidates": [
                {
                    "candidate_index": result["candidate_index"],
                    "prompt": result["prompt"],
                    "score": result["score"],
                    "real_score": result["real_score"],
                    "diversity_score": result["diversity_score"],
                    "distance": result["distance"],
                    "objective": result["objective"],
                    "final_score": result["final_score"],
                }
                for result in candidate_results
            ],
        }
        _save_json(os.path.join(iteration_dir, "candidate_search.json"), candidate_search_payload)

        save_timestamps(best_candidate["prompt"], os.path.join(iteration_dir, "timestamps.txt"), selected_audio_path)

        memory.add(
            best_candidate["prompt"],
            best_candidate["real_score"],
            best_candidate["diversity_score"],
            best_candidate["final_score"],
            selected_audio_path,
            parent_prompt=None if parent_prompt == "NONE" else parent_prompt,
            feature_vector=best_candidate["feature_map"]["feature_values"],
            distance_score=best_candidate["distance"],
            objective_score=best_candidate["objective"],
            scoring_mode=mode_key,
        )
        _write_scores_csv(scores_file, memory)

    plot_path = os.path.join(run_results_dir, "plots", "scores.png")
    _save_plot(plot_path, memory)
    print(f"[*] Saved plot to {plot_path}")

    return memory, run_audio_dir, run_results_dir
