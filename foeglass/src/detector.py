import joblib
import numpy as np

from src.audio_features import build_feature_map, extract_mfcc_mean_std


class Detector:
    def __init__(self, model_path="models/detector.pkl"):
        self.model_path = model_path
        if not self.model_path or not isinstance(self.model_path, str):
            raise ValueError("[!] model_path must be a valid string path.")

        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[!] Detector model not found at {self.model_path}. Train the detector first."
            ) from exc

        print(f"[*] Detector loaded model from {self.model_path}")

    def extract_features(self, audio_path):
        """
        Extracts 13 MFCCs at 16 kHz and concatenates mean and std statistics.
        """
        return extract_mfcc_mean_std(audio_path, sr=16000, n_mfcc=13)

    def extract_feature_map(self, audio_path):
        features = self.extract_features(audio_path)
        return build_feature_map(features)

    def predict_real_score(self, audio_path):
        """
        Returns the probability assigned to the 'real' class.
        """
        features = self.extract_features(audio_path)
        features = features.reshape(1, -1)

        classes = list(self.model.classes_)
        if "real" not in classes:
            raise ValueError("[!] Detector model classes must include 'real'.")

        real_idx = classes.index("real")
        proba = self.model.predict_proba(features)
        return float(proba[0][real_idx])

    def score_with_features(self, audio_path):
        feature_map = self.extract_feature_map(audio_path)
        features = np.array(feature_map["feature_values"], dtype=float).reshape(1, -1)

        classes = list(self.model.classes_)
        if "real" not in classes:
            raise ValueError("[!] Detector model classes must include 'real'.")

        real_idx = classes.index("real")
        proba = self.model.predict_proba(features)
        return float(proba[0][real_idx]), feature_map
