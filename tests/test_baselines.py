import json

from review_distill.baselines import TfidfLogRegBaseline
from review_distill.schema import ASPECT_NAMES, ReviewDiagnosis


def diag(issue: str, contact: bool, **aspects: str) -> ReviewDiagnosis:
    values = dict.fromkeys(ASPECT_NAMES, "not_mentioned") | aspects
    return ReviewDiagnosis.model_validate(
        {"aspects": values, "main_issue": issue, "needs_contact": contact}
    )


LATE = diag("late_delivery", False, delivery_time="negative")
GOOD = diag("none", False, delivery_time="positive")
BROKEN = diag("damaged", True, packaging="negative")

TEXTS = ["chegou muito atrasado", "entrega rápida, adorei", "caixa amassada e produto quebrado"]
LABELS = [LATE, GOOD, BROKEN]


def test_fits_training_data_and_outputs_valid_diagnoses():
    model = TfidfLogRegBaseline()
    model.fit(TEXTS * 5, LABELS * 5)
    preds = model.predict(TEXTS)
    assert preds == LABELS
    assert all(isinstance(p, ReviewDiagnosis) for p in preds)


def test_constant_target_is_predicted_without_training():
    model = TfidfLogRegBaseline()
    model.fit(TEXTS * 5, LABELS * 5)
    # customer_service is not_mentioned everywhere in training
    assert model.models["aspects.customer_service"] == "not_mentioned"
    (pred,) = model.predict(["texto novo qualquer"])
    assert pred.aspects.customer_service == "not_mentioned"


def test_config_is_json_serializable():
    model = TfidfLogRegBaseline()
    model.fit(TEXTS * 5, LABELS * 5)
    config = json.loads(json.dumps(model.config()))
    assert config["seed"] == 42
    assert config["vocabulary_size"] > 0
