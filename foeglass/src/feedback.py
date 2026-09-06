def _build_records(history):
    records = []
    for idx, prompt in enumerate(history["prompts"]):
        records.append(
            {
                "iteration": idx + 1,
                "prompt": prompt,
                "real_score": history["real_scores"][idx],
                "diversity_score": history["diversity_scores"][idx],
                "final_score": history["final_scores"][idx],
                "score_delta": history["score_deltas"][idx],
                "parent_prompt": history["parent_prompts"][idx],
                "feature_vector": history["feature_vectors"][idx],
            }
        )
    return records


def _top_and_bottom_records(history):
    records = _build_records(history)
    records_desc = sorted(records, key=lambda item: item["final_score"], reverse=True)
    records_asc = sorted(records, key=lambda item: item["final_score"])
    return records_desc[:3], records_asc[:3]


def _top_delta_records(history):
    records = _build_records(history)
    return sorted(records, key=lambda item: item["score_delta"], reverse=True)[:3]


def _non_improvement_streak(history):
    final_scores = history["final_scores"]
    if len(final_scores) < 2:
        return 0

    best_so_far = final_scores[0]
    streak = 0
    for score in final_scores[1:]:
        if score > best_so_far:
            best_so_far = score
            streak = 0
        else:
            streak += 1
    return streak


def _recent_prompt_repeat(history):
    prompts = history["prompts"]
    if len(prompts) < 2:
        return False
    return prompts[-1].strip().lower() == prompts[-2].strip().lower()


def _detect_plateau(history):
    return _non_improvement_streak(history) >= 2 or _recent_prompt_repeat(history)


def _tone_schedule(history):
    tones = ["casual", "reflective", "narrative", "fragmented", "whispery"]
    return tones[len(history["prompts"]) % len(tones)]


def _format_feature_ranges(records):
    feature_vectors = [record["feature_vector"] for record in records if record["feature_vector"]]
    if not feature_vectors:
        return "NONE"

    num_features = len(feature_vectors[0])
    means = []
    for idx in range(num_features):
        values = [vector[idx] for vector in feature_vectors]
        means.append(sum(values) / len(values))

    ranked = sorted(
        enumerate(means, start=1),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:4]
    return ", ".join(f"mfcc_{idx}={value:.3f}" for idx, value in ranked)


def _detector_hints(history):
    records = _build_records(history)
    if len(records) < 3:
        return "favor smoother pacing, longer vowels, and low-interruption phrases"

    top_records = sorted(records, key=lambda item: item["real_score"], reverse=True)[:3]
    bottom_records = sorted(records, key=lambda item: item["real_score"])[:3]

    top_avg_length = sum(len(record["prompt"].split()) for record in top_records) / len(top_records)
    bottom_avg_length = sum(len(record["prompt"].split()) for record in bottom_records) / len(bottom_records)
    top_pause_count = sum(record["prompt"].count(",") + record["prompt"].count("...") for record in top_records)
    bottom_pause_count = sum(record["prompt"].count(",") + record["prompt"].count("...") for record in bottom_records)

    hints = []
    if top_avg_length >= bottom_avg_length:
        hints.append("prefer medium-length lines with drawn-out vowel sounds")
    else:
        hints.append("prefer shorter fragments with lighter phrasing")

    if top_pause_count <= bottom_pause_count:
        hints.append("avoid abrupt punctuation bursts and keep the pacing smooth")
    else:
        hints.append("use gentle pauses, but avoid choppy stop-start punctuation")

    hints.append(
        f"high-score MFCC signature: {_format_feature_ranges(top_records)} | low-score MFCC signature: {_format_feature_ranges(bottom_records)}"
    )
    return " ; ".join(hints)


def generate_feedback_context(memory):
    """
    Build detector-aware feedback for the next prompt generation pass.
    """
    history = memory.get_all()
    if not history["prompts"]:
        return (
            "BEST_PROMPT: NONE\n"
            "LAST_PROMPT: NONE\n"
            "ANTI_PLATEAU_MODE: OFF\n"
            "NON_IMPROVEMENT_STREAK: 0\n"
            "TARGET_TONE: casual\n"
            "DETECTOR_HINTS: favor smoother pacing, longer vowels, and soft consonant-heavy words\n"
            "STYLE_TARGETS: detector-aware search | phonetic variation | temporal variation | structural variation | strict diversity\n"
            "INSTRUCTION: Generate short natural English utterances for spoken audio."
        )

    top_3, bottom_3 = _top_and_bottom_records(history)
    top_deltas = _top_delta_records(history)
    best_prompt = top_3[0]["prompt"]
    last_prompt = history["prompts"][-1]
    plateau = _detect_plateau(history)
    streak = _non_improvement_streak(history)
    target_tone = _tone_schedule(history)

    lines = [
        f"BEST_PROMPT: {best_prompt}",
        f"LAST_PROMPT: {last_prompt}",
        f"ANTI_PLATEAU_MODE: {'ON' if plateau else 'OFF'}",
        f"NON_IMPROVEMENT_STREAK: {streak}",
        f"TARGET_TONE: {target_tone}",
        f"DETECTOR_HINTS: {_detector_hints(history)}",
        "STYLE_TARGETS: detector-aware search | phonetic variation | temporal variation | structural variation | strict diversity",
        "TOP_PROMPTS:",
    ]

    for record in top_3:
        lines.append(
            f"- iter={record['iteration']} | prompt={record['prompt']} | final={record['final_score']:.4f} | real={record['real_score']:.4f} | diversity={record['diversity_score']:.4f} | delta={record['score_delta']:.4f}"
        )

    lines.append("TOP_DELTA_PROMPTS:")
    for record in top_deltas:
        lines.append(
            f"- iter={record['iteration']} | prompt={record['prompt']} | delta={record['score_delta']:.4f} | final={record['final_score']:.4f}"
        )

    lines.append("WORST_PROMPTS:")
    for record in bottom_3:
        lines.append(
            f"- iter={record['iteration']} | prompt={record['prompt']} | final={record['final_score']:.4f} | real={record['real_score']:.4f} | diversity={record['diversity_score']:.4f} | delta={record['score_delta']:.4f}"
        )

    lines.extend(
        [
            "INSTRUCTION: Use the BEST_PROMPT as the base pattern, but do not copy it.",
            "INSTRUCTION: Prefer prompts that could improve detector-facing acoustics, not just semantics.",
            "INSTRUCTION: Use drawn-out vowels, soft consonant-heavy words, and smooth pacing when detector hints support them.",
            "INSTRUCTION: Keep prompts natural, short, and suitable for spoken audio.",
            "INSTRUCTION: Avoid meta text, labels, numbers, and repeated prompts.",
        ]
    )

    if plateau:
        lines.append(
            "INSTRUCTION: Anti-plateau mode is ON. Force a structural shift in tone, pacing, and word distribution away from the last prompts."
        )

    return "\n".join(lines)
