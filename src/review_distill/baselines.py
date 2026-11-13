"""TF-IDF + logistic regression baseline.

One shared TF-IDF vectorizer and one independent logistic regression per output field
(six aspects, main_issue, needs_contact). Its output always fits the schema, so it is
100% "valid JSON" by construction. Hyperparameters are fixed (not tuned on the gold set).
"""

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from review_distill.sampling import SEED
from review_distill.schema import ASPECT_NAMES, ReviewDiagnosis

TFIDF_PARAMS: dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
LOGREG_PARAMS: dict[str, Any] = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000}

TARGETS = [*(f"aspects.{name}" for name in ASPECT_NAMES), "main_issue", "needs_contact"]


def _target_values(diagnoses: Sequence[ReviewDiagnosis], target: str) -> list[Any]:
    data = [d.model_dump(mode="json") for d in diagnoses]
    if target.startswith("aspects."):
        return [d["aspects"][target.removeprefix("aspects.")] for d in data]
    return [d[target] for d in data]


class TfidfLogRegBaseline:
    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self.vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        self.models: dict[str, Any] = {}

    def fit(self, texts: Sequence[str], diagnoses: Sequence[ReviewDiagnosis]) -> None:
        x = self.vectorizer.fit_transform(texts)
        for target in TARGETS:
            y = _target_values(diagnoses, target)
            if len(set(y)) == 1:  # nothing to learn: always predict the only value
                self.models[target] = y[0]
                continue
            model = LogisticRegression(**LOGREG_PARAMS, random_state=self.seed)
            self.models[target] = model.fit(x, y)

    def predict(self, texts: Sequence[str]) -> list[ReviewDiagnosis]:
        x = self.vectorizer.transform(texts)
        columns = {}
        for target, model in self.models.items():
            if isinstance(model, LogisticRegression):
                columns[target] = model.predict(x).tolist()
            else:
                columns[target] = [model] * len(texts)
        return [
            ReviewDiagnosis.model_validate(
                {
                    "aspects": {n: columns[f"aspects.{n}"][i] for n in ASPECT_NAMES},
                    "main_issue": columns["main_issue"][i],
                    "needs_contact": bool(columns["needs_contact"][i]),
                }
            )
            for i in range(len(texts))
        ]

    def config(self) -> dict[str, Any]:
        tfidf = {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])}
        return {
            "tfidf": tfidf,
            "logreg": LOGREG_PARAMS,
            "seed": self.seed,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
        }
