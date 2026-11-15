"""Make the README figures from results/*.json and the teacher-labeled pool.

1. results/figures/model_comparison.png: the four models on the gold set.
2. results/figures/contact_by_score.png: share of reviews that need contact, by star rating.
3. results/figures/main_issues.png: what customers complain about.

Figures 2 and 3 use the teacher labels of the 2,000 pool reviews (400 per star rating).
Figure 3 reweights them by the real distribution of ratings among the reviews with text,
so the shares estimate the whole dataset rather than the stratified sample.

Example:
    uv run python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from review_distill.data import load_reviews

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
BLUE = SERIES[0]

MODELS = [
    ("gemini-teacher", "Gemini (professor)"),
    ("qwen2.5-1.5b-lora", "Qwen 1,5B + LoRA"),
    ("qwen2.5-0.5b-lora", "Qwen 0,5B + LoRA"),
    ("tfidf-logreg", "TF-IDF + reg. logística"),
]
METRICS = [
    ("aspects_macro_f1", "Aspectos\n(macro-F1)"),
    ("main_issue_accuracy", "Problema principal\n(acurácia)"),
    ("main_issue_macro_f1", "Problema principal\n(macro-F1)"),
    ("needs_contact_f1", "Precisa de contato\n(F1)"),
]
ISSUES = {
    "late_delivery": "Entrega atrasada",
    "not_delivered": "Não entregue",
    "wrong_item": "Item errado",
    "missing_item": "Item faltando",
    "damaged": "Danificado",
    "poor_quality": "Baixa qualidade / defeito",
    "not_as_described": "Diferente do anúncio",
    "bad_service": "Atendimento ruim",
    "other": "Outro",
}


def style(ax: plt.Axes, axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig: plt.Figure, text: str) -> None:
    """Left-aligned title at the figure edge (not the axes edge, which moves with labels)."""
    fig.suptitle(text, x=0.01, ha="left", color=TEXT, fontsize=12)


def pct(x: float) -> str:
    return f"{x:.0%}".replace(".", ",")


def dec(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def model_comparison() -> Path:
    metrics = {
        name: json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))["metrics"]
        for name, _ in MODELS
    }
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    style(ax, axis="y")
    n = len(MODELS)
    width = 0.8 / n
    for i, (name, label) in enumerate(MODELS):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(METRICS))]
        ys = [metrics[name][key] for key, _ in METRICS]
        ax.bar(xs, ys, width=width - 0.02, color=SERIES[i], label=label, zorder=2)
        if name == "qwen2.5-1.5b-lora":  # label only the final model
            for x, y in zip(xs, ys, strict=True):
                ax.text(x, y + 0.012, dec(y), ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(METRICS)), [label for _, label in METRICS], color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: dec(v))
    ax.set_ylabel("Pontuação no teste ouro (150 avaliações)", color=TEXT_2)
    ax.set_title(
        "O aluno de 1,5B chega a ~90% do professor e supera o TF-IDF em todas as métricas",
        loc="left",
        color=TEXT,
        fontsize=12,
        pad=28,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        handlelength=1,
        borderaxespad=0.2,
    )
    return save(fig, "model_comparison.png")


def pool_labels() -> pd.DataFrame:
    labels = []
    for split in ("train", "val"):
        path = ROOT / "data" / "labeled" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            labels += [json.loads(line) for line in f]
    pool = pd.read_csv(ROOT / "data" / "labeled" / "pool_reviews.csv")
    df = pd.DataFrame(labels).merge(pool[["review_id", "score"]], on="review_id")
    assert len(df) == len(labels), "every labeled review must have a score"
    return df


def contact_by_score(df: pd.DataFrame) -> Path:
    share = df.groupby("score")["needs_contact"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax, axis="y")
    ax.bar([f"{s} ★" for s in share.index], share.values, color=BLUE, width=0.6, zorder=2)
    for i, v in enumerate(share.values):
        ax.text(i, v + 0.015, pct(v), ha="center", va="bottom", fontsize=10, color=TEXT)
    ax.set_ylim(0, max(share.values) * 1.18)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_xlabel("Nota dada pelo cliente", color=TEXT_2)
    title(fig, "A nota não basta: 40% das avaliações de 3 estrelas pedem contato")
    fig.text(
        0.01,
        0.01,
        "Avaliações que precisam de contato, por nota. 2.000 avaliações "
        "rotuladas pelo professor (400 por nota).",
        color=TEXT_2,
        fontsize=8,
    )
    return save(fig, "contact_by_score.png")


def main_issues(df: pd.DataFrame) -> Path:
    population = load_reviews()["score"].value_counts(normalize=True)
    sample = df["score"].value_counts(normalize=True)
    weights = df["score"].map(population / sample)
    shares = weights.groupby(df["main_issue"]).sum() / weights.sum()
    no_problem = shares.get("none", 0.0)
    problems = shares.drop("none").sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax, axis="x")
    labels = [ISSUES[k] for k in problems.index]
    ax.barh(labels, problems.values, color=BLUE, height=0.6, zorder=2)
    for i, v in enumerate(problems.values):
        ax.text(v + 0.003, i, pct(v), va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, problems.max() * 1.15)
    ax.xaxis.set_major_formatter(lambda v, _: pct(v))
    ax.tick_params(axis="y", colors=TEXT)
    title(fig, "Do que os clientes reclamam: entrega e pedido incompleto lideram")
    fig.text(
        0.01,
        0.01,
        f"Problema principal entre as avaliações com texto (estimativa; {pct(no_problem)} "
        "não relatam problema). Rótulos do professor, reponderados pela distribuição real "
        "das notas.",
        color=TEXT_2,
        fontsize=8,
        wrap=True,
    )
    return save(fig, "main_issues.png")


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    df = pool_labels()
    for path in (model_comparison(), contact_by_score(df), main_issues(df)):
        print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
