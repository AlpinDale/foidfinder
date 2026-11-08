"""Turns a tagger's probabilities into the API result (shared by every model and backend)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CANDIDATES = 3  # best below-threshold guesses returned per identity category
CANDIDATE_FLOOR = 0.05
IDENTITY = ("character", "copyright", "artist")
LIMIT = {"year": 2}  # Haruka's era tags are overlapping 3-year windows; as its reference tool, keep the best two


class TagIndex:
    def __init__(self, config: dict):
        self.tags: list[str] = config["tags"]
        if "categories" in config:  # Haruka: one category per tag, in any order
            self.name = config["model"]["name"]
            category = config["categories"]
            self.thresholds: dict[str, float] = config["thresholds"]
        else:  # PixAI: categories as consecutive runs
            self.name = "PixAI Tagger v1.0"
            category = [c for c, n in config["tags_split"] for _ in range(n)]
            self.thresholds = config["category_best_threshold"]
        if len(self.tags) != len(category):
            raise ValueError("tag categories do not cover the tag list")
        cats = np.array(category)
        self.index = {c: np.flatnonzero(cats == c) for c in dict.fromkeys(category)}

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> TagIndex:
        return cls(json.loads((Path(model_dir) / "config.json").read_text()))

    def _entries(self, probs: np.ndarray, ids: np.ndarray) -> list[dict]:
        return [{"tag": self.tags[j], "score": round(float(probs[j]), 4)} for j in ids]

    def result(self, probs: np.ndarray) -> dict:
        """probs: [num_tags] -> {"rating", "tags": {category: [...]}, "candidates": {identity category: [...]}}.

        Tags are sorted by score; a category's list holds the tags at or above its threshold. The rating is the
        most likely rating tag. Candidates are the best few below-threshold characters, series and artists."""
        tags, candidates, rating = {}, {}, None
        for c, idx in self.index.items():
            p = probs[idx]
            if c == "rating":
                j = idx[int(np.argmax(p))]
                rating = {"tag": self.tags[j], "score": round(float(probs[j]), 4)}
                continue
            accepted = idx[p >= self.thresholds[c]]
            tags[c] = self._entries(probs, accepted[np.argsort(-probs[accepted], kind="stable")][: LIMIT.get(c)])
            if c in IDENTITY:
                rest = idx[(p < self.thresholds[c]) & (p >= CANDIDATE_FLOOR)]
                rest = rest[np.argsort(-probs[rest], kind="stable")[:CANDIDATES]]
                candidates[c] = self._entries(probs, rest)
        return {"rating": rating, "tags": tags, "candidates": candidates}
