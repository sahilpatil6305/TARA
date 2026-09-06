import librosa
import numpy as np


def extract_mfcc_mean_std(audio_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_std = np.std(mfccs.T, axis=0)
    return np.concatenate([mfcc_mean, mfcc_std])


def build_feature_map(feature_vector):
    half = len(feature_vector) // 2
    feature_names = [f"mfcc_mean_{idx}" for idx in range(1, half + 1)]
    feature_names.extend(f"mfcc_std_{idx}" for idx in range(1, half + 1))
    return {
        "feature_names": feature_names,
        "feature_values": [float(value) for value in feature_vector],
    }
