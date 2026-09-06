"""
src/llm_generator.py
====================
Prompt generators for the DRAO adversarial loop.
"""
from __future__ import annotations

import json
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

try:
    from groq import Groq
except ImportError:  # pragma: no cover - exercised only in environments without groq
    Groq = None  # type: ignore[assignment]


@dataclass
class PromptRecord:
    """A (transcript, objective_value) pair stored in memory."""

    transcript: str
    detector_score: float
    objective: float
    iteration: int


class PromptMemory:
    """Fixed-size memory that retains the top-K prompt records by objective value."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._records: List[PromptRecord] = []

    def add(
        self,
        transcript: str,
        detector_score: float,
        objective: float,
        iteration: int,
    ) -> None:
        self._records.append(
            PromptRecord(
                transcript=transcript,
                detector_score=detector_score,
                objective=objective,
                iteration=iteration,
            )
        )
        self._records.sort(key=lambda r: r.objective, reverse=True)
        self._records = self._records[: self.top_k]

    def to_string(self) -> str:
        if not self._records:
            return "No successful prompts recorded yet."
        lines = ["Top-performing transcripts so far (descending by objective):"]
        for i, rec in enumerate(self._records, 1):
            short = textwrap.shorten(rec.transcript, width=80, placeholder="...")
            lines.append(
                f"  {i}. [iter={rec.iteration} score={rec.detector_score:.4f} J={rec.objective:+.4f}] {short}"
            )
        return "\n".join(lines)

    def best_objective(self) -> Optional[float]:
        return self._records[0].objective if self._records else None

    def top_prompt(self) -> Optional[PromptRecord]:
        return self._records[0] if self._records else None

    def has_elites(self) -> bool:
        return bool(self._records)

    def records(self) -> List[PromptRecord]:
        return list(self._records)


_SYSTEM_PROMPT = """\
You are an expert adversarial speech synthesis researcher.
Your task is to craft a natural-sounding English transcript that,
when read aloud by a text-to-speech system, is likely to be
classified as genuine human speech by an audio deepfake detector.

Guiding principles:
  - Use natural, conversational language with varied sentence rhythm.
  - Include prosodic diversity: questions, exclamations, short and long phrases.
  - Avoid lists, bullet points, digits, abbreviations, or unusual symbols.
  - Each transcript must be DIFFERENT from previous attempts.
  - Length: 1-4 sentences, 40-120 words.

You will be given a history of the best-performing transcripts.
Analyse what made them successful and innovate beyond them.

Output ONLY the transcript text - no preamble, no quotes, no labels.
"""


def _clean_generated_transcript(raw: str, max_chars: int) -> str:
    text = raw.strip()

    quoted_blocks = re.findall(r'"([^"\n]{20,})"', text)
    if quoted_blocks:
        text = max(quoted_blocks, key=len).strip()
    else:
        text = re.sub(
            r"^\s*(here(?:'s| is)|attempt|transcript|output)\b[^:]*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        lines = [line.strip(" -\"'") for line in text.splitlines() if line.strip()]
        text = " ".join(lines)

    text = text.strip(" \"'")
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


class PromptGenerator(Protocol):
    def generate(self, memory: PromptMemory, iteration: int) -> str:
        ...

    def mutate(self, transcript: str, memory: PromptMemory, iteration: int) -> str:
        ...


class GroqPromptGenerator:
    """Groq-backed generator."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama3-8b-8192",
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 120,
        max_chars: int = 280,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        if Groq is None:
            raise RuntimeError(
                "Groq backend requested, but the 'groq' package is not installed."
            )
        self._client: Any = Groq(api_key=api_key)

    def _chat(self, user_msg: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        return _clean_generated_transcript(raw, self.max_chars)

    def generate(self, memory: PromptMemory, iteration: int) -> str:
        user_msg = (
            f"Iteration {iteration}. "
            f"Previous best results:\n{memory.to_string()}\n\n"
            "Generate a new, DISTINCT transcript that you predict will "
            "perform even better. Output only the transcript."
        )
        return self._chat(user_msg)

    def mutate(self, transcript: str, memory: PromptMemory, iteration: int) -> str:
        user_msg = (
            f"Iteration {iteration}. Previous best results:\n{memory.to_string()}\n\n"
            f"Base transcript:\n{transcript}\n\n"
            "Reuse the base idea but slightly modify the wording, tone, rhythm, or add a natural phrase. "
            "Keep it distinct, fluent, and likely to sound human. Output only the mutated transcript."
        )
        return self._chat(user_msg)


class OllamaPromptGenerator:
    """Local Ollama-backed generator using the localhost HTTP API."""

    def __init__(
        self,
        model: str = "llama3.2:latest",
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_chars: int = 280,
        endpoint: str = "http://127.0.0.1:11434/api/generate",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_chars = max_chars
        self.endpoint = endpoint

    def _generate_from_prompt(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Failed to reach local Ollama at 127.0.0.1:11434. "
                "Make sure the Ollama app/service is running."
            ) from exc

        raw = str(body.get("response", ""))
        if not raw:
            raise RuntimeError(f"Ollama returned an empty response for model {self.model!r}")
        return _clean_generated_transcript(raw, self.max_chars)

    def generate(self, memory: PromptMemory, iteration: int) -> str:
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Iteration {iteration}.\n"
            f"Previous best results:\n{memory.to_string()}\n\n"
            "Generate a new, DISTINCT transcript that you predict will "
            "perform even better. Output only the transcript."
        )
        return self._generate_from_prompt(prompt)

    def mutate(self, transcript: str, memory: PromptMemory, iteration: int) -> str:
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Iteration {iteration}.\n"
            f"Previous best results:\n{memory.to_string()}\n\n"
            f"Base transcript:\n{transcript}\n\n"
            "Reuse the base idea but slightly modify the tone, add a phrase, or reword the lines naturally. "
            "Keep it distinct from the base transcript and output only the mutated transcript."
        )
        return self._generate_from_prompt(prompt)


def build_prompt_generator(
    provider: str,
    groq_api_key: Optional[str],
    groq_model: str,
    ollama_model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_chars: int,
) -> PromptGenerator:
    """Factory for the configured prompt generator backend."""

    provider = provider.strip().lower()
    if provider == "ollama":
        return OllamaPromptGenerator(
            model=ollama_model,
            temperature=temperature,
            top_p=top_p,
            max_chars=max_chars,
        )
    if provider == "groq":
        if not groq_api_key:
            raise RuntimeError(
                "LLM provider is set to 'groq' but GROQ_API_KEY is missing."
            )
        return GroqPromptGenerator(
            api_key=groq_api_key,
            model=groq_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_chars=max_chars,
        )
    raise ValueError(f"Unsupported LLM provider: {provider!r}")
