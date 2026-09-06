import os
from typing import Dict, List


FORBIDDEN_DATASET_MARKERS = {
    "metadata.json",
    "timestamps.txt",
}


def verify_dataset_integrity(dataset_root="dataset") -> Dict[str, object]:
    required_dirs = [
        os.path.join(dataset_root, "train", "real"),
        os.path.join(dataset_root, "train", "fake"),
        os.path.join(dataset_root, "eval", "real"),
        os.path.join(dataset_root, "eval", "fake"),
    ]

    missing_dirs = [path for path in required_dirs if not os.path.isdir(path)]
    if missing_dirs:
        raise FileNotFoundError(
            "[!] Dataset structure is incomplete. Missing directories: "
            + ", ".join(missing_dirs)
        )

    suspicious_paths: List[str] = []
    for root, dirs, files in os.walk(dataset_root):
        for directory_name in dirs:
            if directory_name.startswith("run_") or directory_name.startswith("iteration_"):
                suspicious_paths.append(os.path.join(root, directory_name))
        for file_name in files:
            if file_name in FORBIDDEN_DATASET_MARKERS:
                suspicious_paths.append(os.path.join(root, file_name))

    if suspicious_paths:
        raise RuntimeError(
            "[!] Generated-audio artifacts were found inside dataset/. "
            "This violates AGENT_GUIDE.md: "
            + ", ".join(suspicious_paths)
        )

    return {
        "train_real_exists": True,
        "train_fake_exists": True,
        "eval_real_exists": True,
        "eval_fake_exists": True,
        "suspicious_paths": suspicious_paths,
    }
