"""
src/embedding.py
================
wav2vec2-based audio embedding extractor.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from .logging_utils import suppress_external_output


@dataclass(frozen=True)
class EmbeddingOutput:
    global_embedding: torch.Tensor
    frame_embeddings: torch.Tensor


class Wav2VecEmbedder:
    """Wrap the Hugging Face wav2vec2 model for inference-only embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        sample_rate: int = 16_000,
        device: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        model_source = self._resolve_model_source(model_name)
        self.logger.info(
            "Loading wav2vec model from %s on device=%s",
            model_source,
            self.device,
        )
        with suppress_external_output(self.logger, "[wav2vec]"):
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_source,
                local_files_only=True,
            )
            model, loading_info = Wav2Vec2Model.from_pretrained(
                model_source,
                local_files_only=True,
                output_loading_info=True,
            )
            unexpected = sorted(loading_info.get("unexpected_keys", []))
            missing = sorted(loading_info.get("missing_keys", []))
            mismatched = list(loading_info.get("mismatched_keys", []))
            if missing or mismatched:
                self.logger.warning(
                    "wav2vec load anomalies: missing=%d mismatched=%d",
                    len(missing),
                    len(mismatched),
                )
            elif unexpected:
                self.logger.debug(
                    "wav2vec ignored %d unexpected checkpoint keys from pretraining head.",
                    len(unexpected),
                )
            self.model = model.to(self.device)
        self.model.eval()
        self._logged = 0

    def _resolve_model_source(self, model_name: str) -> str:
        if Path(model_name).exists():
            return model_name

        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = hub_root / f"models--{model_name.replace('/', '--')}"
        ref_file = model_dir / "refs" / "main"
        if ref_file.exists():
            revision = ref_file.read_text(encoding="utf-8").strip()
            snapshot_dir = model_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)

        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
            if snapshots:
                return str(snapshots[-1])

        raise FileNotFoundError(
            f"No local Hugging Face snapshot found for {model_name!r}. "
            "Download it once in an environment with internet access or point "
            "the config to a local model directory."
        )

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        waveform, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        return torch.from_numpy(waveform).view(1, -1)

    @torch.inference_mode()
    def extract_from_waveform(self, waveform: torch.Tensor) -> EmbeddingOutput:
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected waveform shape (1, T), got {tuple(waveform.shape)}"
            )
        inputs = self.processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        hidden = self.model(input_values).last_hidden_state.squeeze(0)
        global_emb = hidden.mean(dim=0)
        return EmbeddingOutput(global_embedding=global_emb, frame_embeddings=hidden)

    @torch.inference_mode()
    def extract(self, audio_path: Path) -> EmbeddingOutput:
        waveform = self.load_audio(audio_path)
        out = self.extract_from_waveform(waveform)
        if self._logged < 3:
            self.logger.info(
                "[embedding] %s waveform=%s global=%s",
                audio_path.name,
                tuple(waveform.shape),
                tuple(out.global_embedding.shape),
            )
            self._logged += 1
        return out

    @torch.inference_mode()
    def batch_extract_global(
        self, audio_paths: Iterable[Path]
    ) -> np.ndarray:
        audio_paths = list(audio_paths)
        total = len(audio_paths)
        embeddings: List[np.ndarray] = []
        for index, path in enumerate(audio_paths, start=1):
            out = self.extract(path)
            embeddings.append(out.global_embedding.cpu().numpy())
            if index == total or index % 100 == 0:
                self.logger.info(
                    "Embedding progress: %d/%d files (%.1f%%)",
                    index,
                    total,
                    100.0 * index / max(total, 1),
                )
        if not embeddings:
            raise ValueError("No audio files supplied for embedding extraction.")
        return np.stack(embeddings, axis=0)


def quick_load_audio(audio_path: Path, sample_rate: int = 16_000) -> np.ndarray:
    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 1.0
    return waveform / max(peak, 1e-8)
