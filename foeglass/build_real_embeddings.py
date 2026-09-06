import argparse
import glob
import os

import numpy as np

from src.config import REAL_EMBEDDINGS_PATH, REAL_TRAIN_DIR
from src.embedding import get_embedding


def build_real_embeddings(real_dir=REAL_TRAIN_DIR, output_path=REAL_EMBEDDINGS_PATH, force=False):
    if os.path.exists(output_path) and not force:
        embeddings = np.load(output_path)
        print(f"[*] Using cached real embeddings from {output_path} with shape {embeddings.shape}")
        return embeddings

    audio_paths = sorted(glob.glob(os.path.join(real_dir, "**", "*.wav"), recursive=True))
    if not audio_paths:
        raise RuntimeError(f"[!] No .wav files found in {real_dir}")

    print(f"[*] Building real embedding bank from {real_dir}")
    embeddings = []
    for idx, audio_path in enumerate(audio_paths, start=1):
        print(f"[*] [{idx}/{len(audio_paths)}] {os.path.basename(audio_path)}")
        embeddings.append(get_embedding(audio_path))

    embedding_matrix = np.stack(embeddings).astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding_matrix)
    print(f"[*] Saved real embeddings to {output_path} with shape {embedding_matrix.shape}")
    return embedding_matrix


def main():
    parser = argparse.ArgumentParser(description="Build and cache wav2vec embeddings for real training audio.")
    parser.add_argument("--real-dir", default=REAL_TRAIN_DIR, help="Directory containing real training .wav files.")
    parser.add_argument("--output", default=REAL_EMBEDDINGS_PATH, help="Output .npy path for cached embeddings.")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even if the cache exists.")
    args = parser.parse_args()

    build_real_embeddings(real_dir=args.real_dir, output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
