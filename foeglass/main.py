import datetime
import json
import os
import sys

# Ensure proper module paths are picked up from the local dir
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DRAO_LAMBDA
from src.compliance import verify_dataset_integrity
from src.run_logging import finalize_run_logging, log_unhandled_exception, setup_run_logging


def build_run_summary(memory, eval_metrics, generated_metrics, run_results_dir, mode, seed):
    history = memory.get_all()
    if not history["final_scores"]:
        raise RuntimeError("[!] No scores were produced by the pipeline.")

    best_idx = history["final_scores"].index(max(history["final_scores"]))
    best_score = history["final_scores"][best_idx]
    avg_score = sum(history["final_scores"]) / len(history["final_scores"])
    baseline_score = history["final_scores"][0]
    final_iteration_score = history["final_scores"][-1]
    baseline_final_improvement = final_iteration_score - baseline_score
    baseline_best_improvement = best_score - baseline_score
    improved_over_iterations = final_iteration_score > baseline_score

    summary = {
        "mode": mode,
        "seed": seed,
        "best_score": best_score,
        "avg_score": avg_score,
        "best_prompt": history["prompts"][best_idx],
        "ASR": generated_metrics["asr"],
        "FNR": eval_metrics["fnr"],
        "baseline_score": baseline_score,
        "final_iteration_score": final_iteration_score,
        "baseline_final_improvement": baseline_final_improvement,
        "baseline_best_improvement": baseline_best_improvement,
        "improved_over_iterations": improved_over_iterations,
        "final_scores": history["final_scores"],
        "detector_scores": history["real_scores"],
        "distance_scores": history["distance_scores"],
        "objective_scores": history["objective_scores"],
        "drao_lambda": DRAO_LAMBDA if mode.lower() == "drao" else None,
    }

    summary_path = os.path.join(run_results_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print(f"[*] Saved run summary to {summary_path}")
    return summary


def execute_run(mode, seed, iterations, validation_dir):
    from evaluation import analyze_run_features, compare_generated_thresholds, evaluate_eval_set, evaluate_generated
    from src.pipeline import run_pipeline

    memory, run_audio_dir, run_results_dir = run_pipeline(
        iterations=iterations,
        candidates_per_iteration=4,
        mode=mode,
        seed=seed,
        drao_lambda=DRAO_LAMBDA,
    )

    eval_metrics = evaluate_eval_set(run_results_dir=run_results_dir, validation_dir=validation_dir)
    generated_metrics = evaluate_generated(run_audio_dir, run_results_dir)
    asr_comparison = compare_generated_thresholds(run_audio_dir, run_results_dir, 0.5, 0.8)
    feature_analysis = analyze_run_features(run_audio_dir, run_results_dir)
    summary = build_run_summary(memory, eval_metrics, generated_metrics, run_results_dir, mode=mode, seed=seed)

    return {
        "memory": memory,
        "run_audio_dir": run_audio_dir,
        "run_results_dir": run_results_dir,
        "eval_metrics": eval_metrics,
        "generated_metrics": generated_metrics,
        "asr_comparison": asr_comparison,
        "feature_analysis": feature_analysis,
        "summary": summary,
    }


def main():
    from evaluation import aggregate_experiment_results, save_baseline_comparison
    from statistical_analysis import run_statistical_analysis
    from train_detector import train

    print("[*] === Starting Adversarial Audio Deepfake Framework ===")

    integrity = verify_dataset_integrity()
    print(
        "[*] Verified dataset integrity: "
        f"train/real={integrity['train_real_exists']}, "
        f"train/fake={integrity['train_fake_exists']}, "
        f"eval/real={integrity['eval_real_exists']}, "
        f"eval/fake={integrity['eval_fake_exists']}"
    )

    validation_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_dir = os.path.join("results", f"validation_{validation_id}")
    os.makedirs(validation_dir, exist_ok=True)
    print(f"[*] Validation artifacts will be saved to: {validation_dir}")

    print("[*] Retraining detector for research-grade validation...")
    os.makedirs("models", exist_ok=True)
    training_results = train(output_dir=validation_dir)

    iterations = 15
    seeds = [42, 84, 126]
    baseline_runs = []
    optimized_runs = []
    baseline_scores = []
    optimized_scores = []
    primary_optimized = None

    for seed in seeds:
        print("\n" + "=" * 50)
        print(f"[*] === Seed {seed}: Baseline Run ===")
        print("=" * 50)
        baseline_run = execute_run(mode="baseline", seed=seed, iterations=iterations, validation_dir=validation_dir)
        baseline_summary = baseline_run["summary"]
        baseline_metrics = baseline_run["generated_metrics"]
        baseline_runs.append(
            {
                "seed": seed,
                "run_results_dir": baseline_run["run_results_dir"],
                "best_score": baseline_summary["best_score"],
                "avg_score": baseline_summary["avg_score"],
                "asr": baseline_metrics["asr"],
            }
        )
        baseline_scores.append(baseline_summary["avg_score"])

        print("\n" + "=" * 50)
        print(f"[*] === Seed {seed}: Optimized Run ===")
        print("=" * 50)
        optimized_run = execute_run(mode="optimized", seed=seed, iterations=iterations, validation_dir=validation_dir)
        optimized_summary = optimized_run["summary"]
        optimized_metrics = optimized_run["generated_metrics"]
        optimized_runs.append(
            {
                "seed": seed,
                "run_results_dir": optimized_run["run_results_dir"],
                "best_score": optimized_summary["best_score"],
                "avg_score": optimized_summary["avg_score"],
                "asr": optimized_metrics["asr"],
            }
        )
        optimized_scores.append(optimized_summary["avg_score"])

        comparison = save_baseline_comparison(
            baseline_metrics,
            optimized_metrics,
            os.path.join(optimized_run["run_results_dir"], "baseline_comparison.json"),
        )
        print(
            "[*] Baseline vs Optimized | "
            f"ASR: {comparison['baseline_asr']:.2f}% -> {comparison['optimized_asr']:.2f}% | "
            f"Avg Final Score: {comparison['baseline_avg_score']:.4f} -> {comparison['optimized_avg_score']:.4f}"
        )

        if seed == seeds[0]:
            primary_optimized = optimized_run

    aggregate_results = aggregate_experiment_results(
        baseline_runs,
        optimized_runs,
        os.path.join(validation_dir, "aggregate_results.json"),
    )

    multi_run_results = {
        "seeds": seeds,
        "baseline_scores": baseline_scores,
        "optimized_scores": optimized_scores,
        "baseline_runs": baseline_runs,
        "optimized_runs": optimized_runs,
    }
    multi_run_results_path = os.path.join(validation_dir, "multi_run_results.json")
    with open(multi_run_results_path, "w", encoding="utf-8") as handle:
        json.dump(multi_run_results, handle, indent=4)
    print(f"[*] Multi-run raw results saved to {multi_run_results_path}")

    statistical_test = run_statistical_analysis(
        input_path=multi_run_results_path,
        output_path=os.path.join(validation_dir, "statistical_test.json"),
    )

    optimized_best_scores = [row["best_score"] for row in optimized_runs]
    optimized_avg_scores = [row["avg_score"] for row in optimized_runs]
    optimized_asrs = [row["asr"] for row in optimized_runs]
    baseline_avg_scores = [row["avg_score"] for row in baseline_runs]
    baseline_asrs = [row["asr"] for row in baseline_runs]

    print("\n[*] === Research Validation Summary ===")
    print(f"[*] CV ROC-AUC: {training_results['cv_roc_auc_mean']:.4f} ± {training_results['cv_roc_auc_std']:.4f}")
    print(
        f"[*] Optimized ASR vs Baseline ASR: "
        f"{sum(optimized_asrs)/len(optimized_asrs):.2f}% vs {sum(baseline_asrs)/len(baseline_asrs):.2f}%"
    )
    print(
        f"[*] Optimized Avg Score vs Baseline Avg Score: "
        f"{sum(optimized_avg_scores)/len(optimized_avg_scores):.4f} vs "
        f"{sum(baseline_avg_scores)/len(baseline_avg_scores):.4f}"
    )
    print(
        f"[*] Multi-run Best Score: {aggregate_results['optimized']['best_score']['mean']:.4f} ± "
        f"{aggregate_results['optimized']['best_score']['std']:.4f}"
    )
    print(
        f"[*] Multi-run Avg Score: {aggregate_results['optimized']['avg_score']['mean']:.4f} ± "
        f"{aggregate_results['optimized']['avg_score']['std']:.4f}"
    )
    print(
        f"[*] Multi-run ASR: {aggregate_results['optimized']['asr']['mean']:.2f}% ± "
        f"{aggregate_results['optimized']['asr']['std']:.2f}%"
    )
    print(f"[*] Baseline Mean ± Std: {statistical_test['baseline_mean']:.4f} ± {statistical_test['baseline_std']:.4f}")
    print(f"[*] Optimized Mean ± Std: {statistical_test['optimized_mean']:.4f} ± {statistical_test['optimized_std']:.4f}")
    print(f"[*] p-value: {statistical_test['p_value']:.6f}")
    print(f"[*] Best Score Achieved: {max(optimized_best_scores):.4f}")
    print(f"[*] Statistically Significant Improvement: {statistical_test['statistically_significant_at_0_05']}")
    print(f"[*] Validation Results Folder: {validation_dir}")

    if primary_optimized:
        print(f"[*] Primary Optimized Run: {primary_optimized['run_results_dir']}")
        print(f"[*] Primary Optimized Audio: {primary_optimized['run_audio_dir']}")


if __name__ == "__main__":
    log_path, log_file, original_stdout, original_stderr = setup_run_logging()
    exit_code = 0
    try:
        main()
        print(f"[*] Run completed. Log saved to: {log_path}")
    except Exception as exc:
        exit_code = 1
        log_unhandled_exception(exc)
        print(f"[!] Run failed. Log saved to: {log_path}")
    finally:
        finalize_run_logging(log_file, original_stdout, original_stderr)

    sys.exit(exit_code)
