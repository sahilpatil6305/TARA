import os


WAV2VEC_MODEL_NAME = os.getenv("FOEGLASS_WAV2VEC_MODEL", "facebook/wav2vec2-base-960h")
REAL_TRAIN_DIR = os.getenv("FOEGLASS_REAL_TRAIN_DIR", os.path.join("dataset", "train", "real"))
REAL_EMBEDDINGS_PATH = os.getenv("FOEGLASS_REAL_EMBEDDINGS_PATH", os.path.join("data", "real_embeddings.npy"))
DRAO_LAMBDA = float(os.getenv("FOEGLASS_DRAO_LAMBDA", "0.5"))
