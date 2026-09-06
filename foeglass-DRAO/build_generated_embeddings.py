"""
build_generated_embeddings.py
=============================
Build cached generated-speech embeddings from previously synthesized audio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import get_config
from src.embedding import Wav2VecEmbedder
from src.logging_utils import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached generated-speech embeddings from data/audio/"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory under data/audio to restrict embedding collection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings even if the cached file looks newer than the inputs.",
    )
    return parser.parse_args()


def _resolve_audio_root(base_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        return base_dir
    return base_dir / run_name


def _is_up_to_date(audio_root: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    wav_files = [path for path in audio_root.rglob("*.wav") if path.is_file()]
    if not wav_files:
        return False
    out_mtime = out_path.stat().st_mtime
    latest_input_mtime = max(path.stat().st_mtime for path in wav_files)
    return out_mtime >= latest_input_mtime


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    audio_root = _resolve_audio_root(cfg.paths.audio_dir, args.run_name)
    out_path = cfg.paths.generated_embeddings_path
    logger = setup_logger(
        "foeglass.build_generated_embeddings",
        cfg.paths.logs_dir / "build_generated_embeddings.log",
    )

    if not audio_root.exists():
        logger.error(
            "[ERROR] Generated audio directory not found:\n"
            "  %s\n\n"
            "Run `python main.py --iterations 3 --run-name smoke_test_3` first,\n"
            "or point this script at an existing run with `--run-name`.",
            audio_root,
        )
        sys.exit(1)

    wav_files = sorted(path for path in audio_root.rglob("*.wav") if path.is_file())
    if not wav_files:
        logger.error(
            "[ERROR] No WAV files found under:\n"
            "  %s\n\n"
            "Expected generated audio in directories like:\n"
            "  data/audio/run_20260427_021150_seed42/t001.wav\n"
            "  data/audio/smoke_test_3/t001.wav",
            audio_root,
        )
        sys.exit(1)

    if not args.force and _is_up_to_date(audio_root, out_path):
        cached = np.load(str(out_path), mmap_mode="r")
        logger.info(
            "Cached generated embeddings are up to date. Reusing %s with shape=%s. Use --force to rebuild.",
            out_path,
            cached.shape,
        )
        return

    run_dirs = sorted({path.parent.name for path in wav_files})
    logger.info(
        "[build_generated_embeddings] Found %d WAV files across %d run directories in %s",
        len(wav_files),
        len(run_dirs),
        audio_root,
    )

    embedder = Wav2VecEmbedder(
        model_name=cfg.model.embedding_model_name,
        sample_rate=cfg.audio.sample_rate,
    )
    embeddings = embedder.batch_extract_global(wav_files)
    logger.info(
        "[build_generated_embeddings] Embedding matrix shape: %s",
        embeddings.shape,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embeddings)
    logger.info("[build_generated_embeddings] Saved -> %s", out_path)


if __name__ == "__main__":
    main()
