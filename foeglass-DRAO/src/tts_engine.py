"""
src/tts_engine.py
=================
Coqui TTS wrapper for the DRAO pipeline.

Responsibilities
----------------
  1. Synthesise audio from a text transcript.
  2. Peak-normalise the output.
  3. Persist the waveform as a WAV file for embedding extraction.
  4. Return the Path to the saved file.
"""
from __future__ import annotations

import logging
import unicodedata
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from TTS.api import TTS
except ImportError:  # pragma: no cover - exercised only in environments without Coqui TTS
    TTS = None  # type: ignore[assignment]

from .logging_utils import suppress_external_output


LOGGER = logging.getLogger(__name__)


class TTSEngine:
    """Thin wrapper around Coqui TTS for reproducible speech synthesis."""

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        sample_rate: int = 16_000,
        normalize: float = 0.98,
        use_gpu: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.normalize = normalize
        if TTS is None:
            raise RuntimeError(
                "Coqui TTS backend requested, but the 'TTS' package is not installed."
            )
        LOGGER.info("[tts] Loading model: %s", model_name)
        with suppress_external_output(LOGGER, "[coqui-init]"):
            self._tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)

    def synthesise(
        self,
        transcript: str,
        output_dir: Path,
        stem: str | None = None,
    ) -> Path:
        """Synthesise `transcript` and save the result as a WAV file."""

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or uuid.uuid4().hex[:8]
        wav_path = output_dir / f"{stem}.wav"
        cleaned_transcript = self._sanitize_transcript(transcript)

        with suppress_external_output(LOGGER, "[coqui-tts]"):
            wav_list = self._tts.tts(text=cleaned_transcript)
        if not wav_list:
            raise RuntimeError(f"TTS returned empty audio for: {cleaned_transcript!r}")

        waveform = np.asarray(wav_list, dtype=np.float32)
        waveform = self._peak_normalize(waveform)
        sf.write(str(wav_path), waveform, self.sample_rate)
        return wav_path

    def _peak_normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Scale waveform so that its peak absolute value == self.normalize."""

        peak = float(np.max(np.abs(waveform)))
        if peak < 1e-8:
            return waveform
        return waveform * (self.normalize / peak)

    def _sanitize_transcript(self, transcript: str) -> str:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
        }
        cleaned = transcript
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        cleaned = (
            unicodedata.normalize("NFKD", cleaned)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        cleaned = " ".join(cleaned.split())
        if cleaned != transcript:
            LOGGER.info("[tts] Sanitized transcript for synthesis compatibility.")
        return cleaned
