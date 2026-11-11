import pandas as pd

from review_distill.data import build_text, prepare_reviews


def test_build_text() -> None:
    assert build_text("Ótimo", "Chegou\r\nrápido  ") == "Ótimo. Chegou rápido"
    assert build_text(float("nan"), "  só msg ") == "só msg"
    assert build_text(None, float("nan")) == ""
    assert build_text("   ", "") == ""


def test_prepare_reviews_drops_empty_and_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "a", "d"],
            "review_score": [5, 1, 3, 5, 4],
            "review_comment_title": [None, None, None, None, None],
            "review_comment_message": ["bom", None, "ruim", "outro pedido", "bom"],
        }
    )
    out = prepare_reviews(raw)
    assert out["review_id"].tolist() == ["a", "c"]  # b empty, 2nd a repeated id, d same text
    assert out.columns.tolist() == ["review_id", "score", "text"]
