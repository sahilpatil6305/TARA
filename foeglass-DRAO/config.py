"""
config.py
=========
Centralised configuration for the FoeGlass DRAO framework.
All hyper-parameters live here; nothing is hard-coded elsewhere.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass
class ModelConfig:
    """wav2vec-2 embedding + MLP detector hyper-parameters."""

    embedding_model_name: str = "facebook/wav2vec2-base"
    detector_hidden_dims: List[int] = field(default_factory=lambda: [512, 128])
    detector_dropout: float = 0.3
    detector_weight_decay: float = 1e-4
    detector_lr: float = 1e-3
    detector_batch_size: int = 32
    detector_epochs: int = 50
    detector_patience: int = 7


@dataclass
class OptimizationConfig:
    """DRAO objective J_t hyper-parameters."""

    mode: str = "drao"
    iterations: int = 20
    alpha_attack_gain: float = 3.0
    beta_absolute_score: float = 1.0
    margin_weight: float = 0.25
    margin_target: float = 0.2
    lambda_mmd: float = 0.2
    lambda_smoothness: float = 0.1
    buffer_size: int = 20
    memory_top_k: int = 5
    memory_reuse_probability: float = 0.5
    momentum_gamma: float = 0.6
    asr_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])
    seed: int = 42
    initial_mmd_reference: float = 0.0
    use_mmd: bool = True
    use_smoothness: bool = True
    use_logit: bool = False
    use_memory: bool = True
    use_margin: bool = True
    use_delta_f: bool = True
    use_delta_score: bool = True
    use_momentum: bool = True
    normalize_terms: bool = True
    memory_metric: str = "objective"
    exploitation_threshold: float = 0.05
    exploitation_probability: float = 0.5
    adaptive_score_threshold: float = 0.1
    adaptive_decay_factor: float = 0.5


@dataclass
class AudioConfig:
    """TTS synthesis and audio normalisation settings."""

    sample_rate: int = 16_000
    normalize_peak: float = 0.98
    tts_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    max_prompt_chars: int = 280


@dataclass
class LLMConfig:
    """Groq / LLM prompt-generation settings."""

    provider: str = os.getenv("LLM_PROVIDER", "ollama")
    groq_model: str = "llama3-8b-8192"
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 120


@dataclass
class PathsConfig:
    """All filesystem paths derived from ROOT."""

    root: Path = ROOT
    dataset_dir: Path = ROOT / "dataset"
    data_dir: Path = ROOT / "data"
    audio_dir: Path = ROOT / "data" / "audio"
    buffers_dir: Path = ROOT / "data" / "buffers"
    models_dir: Path = ROOT / "models"
    experiments_dir: Path = ROOT / "experiments"
    results_dir: Path = ROOT / "results"
    logs_dir: Path = ROOT / "logs"
    plots_dir: Path = ROOT / "plots"
    dataset_real_train_dir: Path = ROOT / "dataset" / "WAVE"
    real_embeddings_path: Path = ROOT / "data" / "real_embeddings.npy"
    generated_embeddings_path: Path = ROOT / "data" / "generated_embeddings.npy"
    detector_path: Path = ROOT / "models" / "detector.pt"
    detector_a_path: Path = ROOT / "models" / "detector_A.pt"
    detector_b_path: Path = ROOT / "models" / "detector_B.pt"

    def for_session(self, session_root: Path) -> "PathsConfig":
        session_root = Path(session_root)
        return replace(
            self,
            data_dir=session_root / "data",
            audio_dir=session_root / "audio",
            buffers_dir=session_root / "data" / "buffers",
            results_dir=session_root / "results",
            logs_dir=session_root / "logs",
            plots_dir=session_root / "plots",
        )


@dataclass
class ExperimentConfig:
    """Root configuration object passed through the entire pipeline."""

    name: str = "foeglass_drao"
    description: str = (
        "Distribution-Regularised Adversarial Optimisation "
        "for audio deepfake generation"
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["paths"] = {k: str(v) for k, v in payload["paths"].items()}
        return payload

    def save_snapshot(self, target: Path) -> None:
        """Persist a JSON snapshot of the full config beside the run artefacts."""

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def get_config() -> ExperimentConfig:
    """Factory that returns a fresh config instance with defaults."""

    return ExperimentConfig()
