`data/` stores machine-generated experiment artifacts, not source dataset files.

Contents:

`real_embeddings.npy`
Reference embeddings built from `dataset/train/real/`.

`generated_embeddings.npy`
Embeddings collected from generated attack samples.

`audio/run_<timestamp>/`
Synthesized audio and per-iteration metadata.

`buffers/`
Saved rolling generated-embedding buffers used by the MMD objective.
