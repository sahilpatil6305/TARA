import json
import os

import numpy as np
from scipy.stats import ttest_rel


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_statistical_analysis(
    input_path=os.path.join("results", "multi_run_results.json"),
    output_path=os.path.join("results", "statistical_test.json"),
):
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    baseline_scores = [float(value) for value in data["baseline_scores"]]
    optimized_scores = [float(value) for value in data["optimized_scores"]]

    if len(baseline_scores) != len(optimized_scores):
        raise ValueError("[!] baseline_scores and optimized_scores must have the same length for a paired t-test.")
    if len(baseline_scores) < 2:
        raise ValueError("[!] At least two paired runs are required for statistical validation.")

    t_stat, p_value = ttest_rel(optimized_scores, baseline_scores)

    results = {
        "baseline_mean": float(np.mean(baseline_scores)),
        "baseline_std": _sample_std(baseline_scores),
        "optimized_mean": float(np.mean(optimized_scores)),
        "optimized_std": _sample_std(optimized_scores),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "statistically_significant_at_0_05": bool(p_value < 0.05),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    print(f"[*] Baseline Mean ± Std: {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
    print(f"[*] Optimized Mean ± Std: {results['optimized_mean']:.4f} ± {results['optimized_std']:.4f}")
    print(f"[*] t-statistic: {results['t_stat']:.4f}")
    print(f"[*] p-value: {results['p_value']:.6f}")
    print(f"[*] Statistically Significant Improvement: {results['statistically_significant_at_0_05']}")
    print(f"[*] Statistical test saved to {output_path}")

    return results


if __name__ == "__main__":
    run_statistical_analysis()
