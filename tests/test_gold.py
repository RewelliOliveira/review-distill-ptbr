from collections.abc import Iterable

from review_distill.gold import (
    diagnosis_from_record,
    edit_diagnosis,
    format_diagnosis,
    gold_record,
    load_labels,
    save_gold_record,
)
from review_distill.schema import ASPECT_NAMES, ReviewDiagnosis

RECORD = {
    "review_id": "r1",
    "text": "não chegou",
    "aspects": dict.fromkeys(ASPECT_NAMES, "not_mentioned"),
    "main_issue": "none",
    "needs_contact": False,
    "model": "fake-model",
    "prompt_version": "v1",
}


def scripted(answers: Iterable[str]):
    it = iter(answers)
    return lambda _prompt: next(it)


def test_edit_keeps_everything_on_empty_answers():
    d = diagnosis_from_record(RECORD)
    assert edit_diagnosis(d, scripted([""] * 8)) == d


def test_edit_changes_fields():
    d = diagnosis_from_record(RECORD)
    # delivery_time=n, rest of aspects kept, main_issue=2 (not_delivered), needs_contact=y
    answers = ["n", "", "", "", "", "", "2", "y"]
    edited = edit_diagnosis(d, scripted(answers))
    assert edited.aspects.delivery_time == "negative"
    assert edited.aspects.packaging == "not_mentioned"
    assert edited.main_issue == "not_delivered"
    assert edited.needs_contact is True


def test_edit_reasks_on_invalid_answer_and_accepts_issue_name():
    d = diagnosis_from_record(RECORD)
    answers = ["x", "p", "", "", "", "", "", "foo", "damaged", "maybe", "n"]
    edited = edit_diagnosis(d, scripted(answers))
    assert edited.aspects.delivery_time == "positive"
    assert edited.main_issue == "damaged"
    assert edited.needs_contact is False


def test_gold_record_flags_changes_and_roundtrips(tmp_path):
    teacher = diagnosis_from_record(RECORD)
    edited = teacher.model_copy(update={"needs_contact": True})
    path = tmp_path / "gold.jsonl"

    save_gold_record(gold_record("r1", "t", 1, teacher, teacher, RECORD), path)
    assert load_labels(path)["r1"]["changed"] is False

    save_gold_record(gold_record("r1", "t", 1, edited, teacher, RECORD), path)
    latest = load_labels(path)["r1"]
    assert latest["changed"] is True
    assert latest["teacher_model"] == "fake-model"
    assert diagnosis_from_record(latest) == edited
    assert isinstance(diagnosis_from_record(latest), ReviewDiagnosis)


def test_format_diagnosis_lists_all_fields():
    out = format_diagnosis(diagnosis_from_record(RECORD))
    for name in (*ASPECT_NAMES, "main_issue", "needs_contact"):
        assert name in out
