"""
build_real_embeddings.py
========================
Step 1 of the DRAO execution sequence.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import get_config
from src.embedding import Wav2VecEmbedder
from src.logging_utils import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached real-speech embeddings")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings even if the cached file looks up to date.",
    )
    return parser.parse_args()


def _is_up_to_date(dataset_dir: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    out_mtime = out_path.stat().st_mtime
    latest_input_mtime = max(path.stat().st_mtime for path in dataset_dir.rglob("*") if path.is_file())
    return out_mtime >= latest_input_mtime


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    dataset_dir: Path = cfg.paths.dataset_real_train_dir
    out_path: Path = cfg.paths.real_embeddings_path
    logger = setup_logger("foeglass.build_embeddings", cfg.paths.logs_dir / "build_real_embeddings.log")

    wav_files = sorted(dataset_dir.rglob("*.wav"))
    if not wav_files:
        logger.error(
            f"[ERROR] No WAV files found under:\n  {dataset_dir}\n\n"
            "Expected structure:\n"
            "  dataset/WAVE/SPEAKER0001/audio.wav\n"
            "  dataset/WAVE/SPEAKER0003/audio.wav\n"
            "  ..."
        )
        sys.exit(1)

    if not args.force and _is_up_to_date(dataset_dir, out_path):
        cached = np.load(str(out_path), mmap_mode="r")
        logger.info(
            "Cached embeddings are up to date. Reusing %s with shape=%s. Use --force to rebuild.",
            out_path,
            cached.shape,
        )
        return

    speakers = {f.parent for f in wav_files}
    logger.info(
        "[build_real_embeddings] Found %d WAV files across %d speakers in %s",
        len(wav_files),
        len(speakers),
        dataset_dir,
    )

    embedder = Wav2VecEmbedder(
        model_name=cfg.model.embedding_model_name,
        sample_rate=cfg.audio.sample_rate,
    )

    embeddings = embedder.batch_extract_global(wav_files)
    logger.info("[build_real_embeddings] Embedding matrix shape: %s", embeddings.shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embeddings)
    logger.info("[build_real_embeddings] Saved -> %s", out_path)


if __name__ == "__main__":
    main()
