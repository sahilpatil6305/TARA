import os
from functools import lru_cache

import librosa
import numpy as np

from src.config import WAV2VEC_MODEL_NAME


TARGET_SAMPLE_RATE = 16000


def _load_audio(audio_path, sample_rate=TARGET_SAMPLE_RATE):
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


@lru_cache(maxsize=1)
def _load_wav2vec(model_name=WAV2VEC_MODEL_NAME):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("[!] PyTorch is required for wav2vec embedding extraction.") from exc

    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except ImportError as exc:
        raise RuntimeError("[!] transformers is required for wav2vec embedding extraction.") from exc

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"[!] Failed to load pretrained model '{model_name}'. "
            "Ensure the model is cached locally or that compatible torch/transformers versions are installed."
        ) from exc

    model.eval()
    return processor, model, torch


@lru_cache(maxsize=256)
def _get_embedding_cached(audio_path, model_name, modified_time):
    del modified_time
    processor, model, torch = _load_wav2vec(model_name)
    audio = _load_audio(audio_path)
    inputs = processor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)

    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
        pooled = hidden.mean(dim=1).squeeze(0)

    return pooled.cpu().numpy().astype(np.float32)


def get_embedding(audio_path):
    """
    Extract a 1D wav2vec2 embedding using mean pooling over the time dimension.
    """
    resolved_path = os.path.abspath(audio_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"[!] Audio file not found: {resolved_path}")

    return _get_embedding_cached(
        resolved_path,
        WAV2VEC_MODEL_NAME,
        os.path.getmtime(resolved_path),
    ).copy()
