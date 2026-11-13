"""Review the gold set by hand: accept or edit each Gemini label.

Shows each gold review with Gemini's label; reviewed labels are appended to
data/gold/gold_labels.jsonl. Already reviewed reviews are skipped, so you can stop
(q) and resume at any time. Use --redo ID to review one again (the last decision wins).

Keys: Enter = accept, e = edit, s = skip for now, q = quit.
While editing, Enter keeps the current value of a field.

Example:
    uv run python scripts/review_gold.py
"""

import argparse

from review_distill.gold import (
    GEMINI_GOLD_PATH,
    GOLD_LABELS_PATH,
    diagnosis_from_record,
    edit_diagnosis,
    format_diagnosis,
    gold_record,
    load_labels,
    save_gold_record,
)
from review_distill.sampling import GOLD_REVIEWS_PATH, load_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--redo", nargs="+", default=[], metavar="ID", help="review IDs again")
    args = parser.parse_args()

    gold = load_sample(GOLD_REVIEWS_PATH)
    teacher = load_labels(GEMINI_GOLD_PATH)
    reviewed = load_labels(GOLD_LABELS_PATH)

    missing = set(gold["review_id"]) - set(teacher)
    if missing:
        print(f"warning: {len(missing)} gold reviews have no Gemini label yet (skipped)")

    todo = gold[gold["review_id"].isin(teacher)]
    if args.redo:
        todo = todo[todo["review_id"].isin(args.redo)]
    else:
        todo = todo[~todo["review_id"].isin(reviewed)]

    total = len(todo)
    done_before = len(reviewed)
    print(f"{done_before} already reviewed, {total} to go\n")

    for i, row in enumerate(todo.itertuples(index=False), 1):
        record = teacher[row.review_id]
        label = diagnosis_from_record(reviewed.get(row.review_id, record))
        print("=" * 72)
        print(f"[{i}/{total}] {row.review_id}  score={row.score}")
        print(f"\n{row.text}\n")
        print(format_diagnosis(label))
        while True:
            action = input("\n[Enter] accept  [e] edit  [s] skip  [q] quit: ").strip().lower()
            if action in {"", "e", "s", "q"}:
                break
        if action == "q":
            break
        if action == "s":
            continue
        if action == "e":
            label = edit_diagnosis(label, input)
            print("\n" + format_diagnosis(label))
        teacher_label = diagnosis_from_record(record)
        save_gold_record(
            gold_record(row.review_id, row.text, row.score, label, teacher_label, record)
        )

    final = load_labels(GOLD_LABELS_PATH)
    changed = sum(r["changed"] for r in final.values())
    print(f"\nreviewed: {len(final)}/{len(gold)}  changed vs Gemini: {changed}")


if __name__ == "__main__":
    main()
