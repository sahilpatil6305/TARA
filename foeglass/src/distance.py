import numpy as np


def compute_distance(z, real_embeddings):
    """
    Compute the minimum L2 distance between one embedding and a real embedding bank.
    """
    z_array = np.asarray(z, dtype=np.float32).reshape(1, -1)
    real_array = np.asarray(real_embeddings, dtype=np.float32)

    if real_array.ndim != 2 or real_array.shape[0] == 0:
        raise ValueError("[!] real_embeddings must be a non-empty 2D array.")
    if real_array.shape[1] != z_array.shape[1]:
        raise ValueError(
            "[!] Embedding dimension mismatch: "
            f"query={z_array.shape[1]}, real_bank={real_array.shape[1]}"
        )

    distances = np.linalg.norm(real_array - z_array, axis=1)
    return float(np.min(distances))
