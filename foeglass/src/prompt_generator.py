import random
import re
import subprocess
import unicodedata


USE_LLM_CANDIDATE = False

SOFT_WORDS = ["mellow", "softer", "hush", "slowly", "easy", "little", "velvet", "hollow", "low"]
LONG_VOWELS = ["ummm", "sooo", "well", "uh", "mm", "slowly", "easy"]
TONES = ["casual", "reflective", "narrative", "fragmented", "whispery"]
BASELINE_PROMPTS = [
    "The weather today is pleasant and mild.",
    "I walked home after work and made some tea.",
    "The room was quiet while the lights stayed low.",
    "We talked for a while and then the evening ended.",
    "The street outside sounded calm and far away.",
    "I sat by the window and listened to the rain.",
    "The air felt cool and everything seemed still.",
    "A small lamp glowed softly near the chair.",
]


def _sanitize_prompt(output_text):
    normalized = unicodedata.normalize("NFKD", output_text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("\n", " ")
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    ascii_text = re.sub(r"[^A-Za-z0-9 ,.'?!-]", "", ascii_text)

    sentences = re.split(r"(?<=[.!?])\s+", ascii_text)
    sentence = sentences[0].strip() if sentences else ascii_text
    words = sentence.split()

    if len(words) > 18:
        sentence = " ".join(words[:18]).rstrip(",") + "."

    if sentence and sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def _extract_field(context, field_name, default=""):
    pattern = rf"^{field_name}:\s*(.*)$"
    match = re.search(pattern, context, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def _extract_history_prompts(context):
    prompts = re.findall(r"prompt=(.*?) \|", context)
    return [prompt.strip() for prompt in prompts]


def _theme_from_best_prompt(best_prompt):
    lowered = best_prompt.lower()
    if "late" in lowered or "tonight" in lowered:
        return "evening"
    for candidate in ["night", "evening", "room", "window", "air", "rain", "hall", "coffee", "street", "quiet"]:
        if candidate in lowered:
            return candidate
    return "evening"


def _ngram_set(text, n):
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[idx : idx + n]) for idx in range(len(words) - n + 1)}


def _similarity_score(candidate, prompt):
    candidate_words = set(re.findall(r"\w+", candidate.lower()))
    prompt_words = set(re.findall(r"\w+", prompt.lower()))
    if not candidate_words or not prompt_words:
        return 0.0

    word_overlap = len(candidate_words & prompt_words) / max(len(candidate_words | prompt_words), 1)
    trigram_overlap = len(_ngram_set(candidate, 3) & _ngram_set(prompt, 3)) / max(
        len(_ngram_set(candidate, 3) | _ngram_set(prompt, 3)),
        1,
    )
    return 0.65 * word_overlap + 0.35 * trigram_overlap


def _too_similar(candidate, history_prompts):
    for prompt in history_prompts:
        if candidate.strip().lower() == prompt.strip().lower():
            return True
        if _similarity_score(candidate, prompt) > 0.52:
            return True
    return False


def _looks_bad(candidate):
    lowered = candidate.lower()
    forbidden = [
        "score",
        "prompt",
        "instruction",
        "context",
        "generate",
        "utterance",
        "final=",
        "real=",
        "anti_plateau",
        "best_prompt",
        "last_prompt",
        "target_tone",
        "top_prompts",
        "worst_prompts",
        "detector_hints",
        "spoken-audio",
        "detector-faced",
        "topicdescription",
        "iter=",
    ]
    if any(token in lowered for token in forbidden):
        return True
    if len(candidate.split()) < 4:
        return True
    return False


def _llm_candidate(context):
    if not USE_LLM_CANDIDATE:
        return ""

    system_instruction = (
        "Generate one short English spoken-audio utterance. "
        "Use natural detector-aware variation: smoother pacing, soft consonants, longer vowels, "
        "and only light punctuation. Output only the utterance."
    )
    full_prompt = f"{system_instruction}\n\n{context}\n\nUtterance:"

    result = subprocess.run(
        ["ollama", "run", "tinyllama"],
        input=full_prompt,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        timeout=30,
    )
    return _sanitize_prompt(result.stdout.strip())


def _seed_candidates():
    return [
        "Ummm, this evening is soo quiet tonight, kind of gentle all through.",
        "Well, it's getting late, and everything feels pretty quiet right now.",
        "Sooo, the evening settles into a softer hush, and I just stay with it.",
        "Well, this evening keeps going low and easy, with hardly any edge at all.",
    ]


def _best_prompt_variants(best_prompt):
    lowered = best_prompt.lower()
    if "evening" in lowered or "late" in lowered or "tonight" in lowered:
        return [
            "Ummm, this evening is soo quiet tonight, kind of gentle all through.",
            "Well, it's getting late, and everything feels pretty quiet right now.",
            "Sooo, the evening settles into a softer hush, and I just stay with it.",
            "Well, this evening keeps going low and easy, with hardly any edge at all.",
            "Ummm, the evening feels sooo mellow now, slow and steady all the way through.",
        ]
    return []


def _anchor_variants(best_prompt, theme, tone, detector_hints):
    smoother = "smooth" in detector_hints or "longer vowels" in detector_hints
    short_bias = "shorter fragments" in detector_hints

    anchored = _best_prompt_variants(best_prompt)
    casual = [
        f"Well, the {theme} feels a little softer now, easy and low.",
        f"Well, it's getting a little late, and the {theme} feels quiet and easy right now.",
        f"Ummm, this {theme} is soo quiet tonight, kind of gentle all through.",
        f"Well, the {theme} stays mellow now, and everything moves slowly.",
    ]
    reflective = [
        f"Sooo, the {theme} settles into a softer hush, and I just stay with it.",
        f"Well, the {theme} feels low and steady now, almost too calm to disturb.",
        f"Ummm, this {theme} keeps drifting softly, like a quiet thought staying open.",
    ]
    narrative = [
        f"I walked in, and the {theme} was already soft, slow, and easy to follow.",
        f"The {theme} dropped low for a second, then settled into a mellow quiet.",
        f"Well, the {theme} kept moving slowly, like a calm story under its breath.",
    ]
    fragmented = [
        f"Just the {theme}, low and soft... ummm, still there.",
        f"Sooo quiet, that {theme}, just a little hush and then more hush.",
        f"The {theme}... easy, low, kind of drifting.",
    ]
    whispery = [
        f"Ummm, the {theme} is so soft now, almost whisper-thin and steady.",
        f"Well, this {theme} keeps going low and easy, with hardly any edge at all.",
        f"Sooo, the {theme} just hangs there, quiet and little and smooth.",
    ]

    pool = {
        "casual": casual,
        "reflective": reflective,
        "narrative": narrative,
        "fragmented": fragmented,
        "whispery": whispery,
    }.get(tone, casual)
    pool = anchored + pool

    if smoother:
        pool.extend(
            [
                f"Ummm, the {theme} feels sooo mellow now, slow and steady all the way through.",
                f"Well, this {theme} stays easy and soft, with a long quiet drift to it.",
            ]
        )

    if short_bias:
        pool.extend(
            [
                f"The {theme}... low, easy, still.",
                f"Just that {theme}, soft and slow.",
            ]
        )

    return pool


def _drastic_candidates(theme):
    return [
        f"Not really a story... just the {theme}, then a hush, then another little pause.",
        f"I was halfway through a thought, ummm, and the {theme} went low and easy.",
        f"Well... maybe just that {theme}, soft-soft, then nothing sharp after it.",
        f"The {theme} breaks, then settles, then keeps this mellow little drift going.",
        f"Sooo, not a full sentence, just a slow {theme} and a quiet breath under it.",
    ]


def _unique_candidates(candidates):
    seen = set()
    unique = []
    for candidate in candidates:
        cleaned = _sanitize_prompt(candidate)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def generate_candidates(context, num_candidates=4):
    best_prompt = _extract_field(context, "BEST_PROMPT", "NONE")
    tone = _extract_field(context, "TARGET_TONE", "casual")
    plateau = _extract_field(context, "ANTI_PLATEAU_MODE", "OFF") == "ON"
    detector_hints = _extract_field(context, "DETECTOR_HINTS", "")
    history_prompts = _extract_history_prompts(context)

    if best_prompt == "NONE":
        candidates = _seed_candidates()
        return candidates[:num_candidates]

    theme = _theme_from_best_prompt(best_prompt)
    candidates = []

    try:
        llm_candidate = _llm_candidate(context)
        if llm_candidate:
            candidates.append(llm_candidate)
    except Exception as exc:
        print(f"[!] LLM prompt generation failed, falling back to heuristic candidates: {exc}")

    candidates.extend(_anchor_variants(best_prompt, theme, tone, detector_hints))

    if plateau:
        for other_tone in TONES:
            if other_tone != tone:
                candidates.extend(_anchor_variants(best_prompt, theme, other_tone, detector_hints)[:1])
        candidates.extend(_drastic_candidates(theme))

    filtered = []
    for candidate in _unique_candidates(candidates):
        if _looks_bad(candidate):
            continue
        if _too_similar(candidate, history_prompts):
            continue
        filtered.append(candidate)

    if len(filtered) < num_candidates:
        for fallback in _seed_candidates() + _drastic_candidates(theme):
            cleaned = _sanitize_prompt(fallback)
            if cleaned not in filtered and not _looks_bad(cleaned) and not _too_similar(cleaned, history_prompts):
                filtered.append(cleaned)
            if len(filtered) >= num_candidates:
                break

    return filtered[:num_candidates]


def generate_prompt(context):
    """
    Backward-compatible single-prompt helper.
    """
    candidates = generate_candidates(context, num_candidates=1)
    return candidates[0] if candidates else "Well, the evening feels low and soft, then just eases out."


def generate_baseline_prompt(iteration, seed):
    rng = random.Random(seed + iteration)
    prompt = rng.choice(BASELINE_PROMPTS)
    return _sanitize_prompt(prompt)
