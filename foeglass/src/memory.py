class Memory:
    def __init__(self):
        self.history = {
            "prompts": [],
            "real_scores": [],
            "diversity_scores": [],
            "distance_scores": [],
            "objective_scores": [],
            "final_scores": [],
            "score_deltas": [],
            "embeddings": [],
            "audio_paths": [],
            "parent_prompts": [],
            "feature_vectors": [],
            "scoring_modes": [],
        }

    def add(
        self,
        prompt,
        real_score,
        diversity_score,
        final_score,
        audio_path,
        parent_prompt=None,
        feature_vector=None,
        distance_score=None,
        objective_score=None,
        scoring_mode="legacy",
    ):
        previous_best = max(self.history["final_scores"]) if self.history["final_scores"] else None
        score_delta = final_score - previous_best if previous_best is not None else 0.0

        self.history["prompts"].append(prompt)
        self.history["real_scores"].append(real_score)
        self.history["diversity_scores"].append(diversity_score)
        self.history["distance_scores"].append(distance_score)
        self.history["objective_scores"].append(objective_score if objective_score is not None else final_score)
        self.history["final_scores"].append(final_score)
        self.history["score_deltas"].append(score_delta)
        self.history["audio_paths"].append(audio_path)
        self.history["parent_prompts"].append(parent_prompt)
        self.history["feature_vectors"].append(feature_vector if feature_vector is not None else [])
        self.history["scoring_modes"].append(scoring_mode)

        # Diversity currently uses prompt text as the embedding placeholder.
        self.history["embeddings"].append(prompt)

    def get_all(self):
        return self.history
