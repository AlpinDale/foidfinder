"""Turns PixAI's 30877 probabilities into the API result (shared by the MLX and CUDA backends)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CANDIDATES = 3  # below-threshold character/copyright guesses returned
CANDIDATE_FLOOR = 0.05


class TagIndex:
    def __init__(self, config: dict):
        self.tags: list[str] = config["tags"]
        category = [c for c, n in config["tags_split"] for _ in range(n)]
        if not len(self.tags) == len(category) == config["num_classes"]:
            raise ValueError("PixAI tags_split does not cover the tag list")
        self.thresholds: dict[str, float] = config["category_best_threshold"]
        cats = np.array(category)
        self.index = {c: np.flatnonzero(cats == c) for c in dict.fromkeys(category)}

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> TagIndex:
        return cls(json.loads((Path(model_dir) / "config.json").read_text()))

    def _entries(self, probs: np.ndarray, ids: np.ndarray) -> list[dict]:
        return [{"tag": self.tags[j], "score": round(float(probs[j]), 4)} for j in ids]

    def result(self, probs: np.ndarray) -> dict:
        """probs: [num_tags] -> {"rating", "tags": {category: [...]}, "candidates": {character|copyright: [...]}}.

        Tags are sorted by score; a category's list holds the tags at or above PixAI's threshold for it.
        The rating is the most likely of the four rating tags."""
        tags, candidates, rating = {}, {}, None
        for c, idx in self.index.items():
            p = probs[idx]
            if c == "rating":
                j = idx[int(np.argmax(p))]
                rating = {"tag": self.tags[j], "score": round(float(probs[j]), 4)}
                continue
            accepted = idx[p >= self.thresholds[c]]
            tags[c] = self._entries(probs, accepted[np.argsort(-probs[accepted], kind="stable")])
            if c in ("character", "copyright"):
                rest = idx[(p < self.thresholds[c]) & (p >= CANDIDATE_FLOOR)]
                rest = rest[np.argsort(-probs[rest], kind="stable")[:CANDIDATES]]
                candidates[c] = self._entries(probs, rest)
        return {"rating": rating, "tags": tags, "candidates": candidates}
