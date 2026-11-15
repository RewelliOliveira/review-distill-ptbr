"""Publish the student model, the labeled dataset and the demo Space to the Hugging Face Hub.

Meant to run on Colab (the trained adapter lives in Google Drive) with HF_TOKEN set
(write access). Every repo gets a card generated from results/*.json, so the numbers on
the Hub are the same as in the README. Use --dry-run to only build the folders locally.

Example (Colab, after cloning the repo and mounting Drive):
    python scripts/publish_hf.py --user Rewelli \\
        --adapter-dir /content/drive/MyDrive/review-distill/qwen2.5-1.5b-lora/adapter \\
        --data-dir /content/drive/MyDrive/review-distill
"""

import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB_URL = "https://github.com/RewelliOliveira/review-distill-ptbr"
COLAB_DEMO_URL = (
    "https://colab.research.google.com/github/RewelliOliveira/review-distill-ptbr/"
    "blob/main/notebooks/03_demo.ipynb"
)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "review-distill-qwen2.5-1.5b-lora"
DATASET_NAME = "olist-reviews-diagnosis-ptbr"
SPACE_NAME = "review-distill-demo"
TEACHER = "Gemini 3.5 Flash-Lite, prompt v3"

RESULT_ROWS = [
    ("gemini-teacher", "Gemini 3.5 Flash-Lite (professor)"),
    ("qwen2.5-1.5b-lora", "**Qwen2.5-1.5B + LoRA (este modelo)**"),
    ("qwen2.5-0.5b-lora", "Qwen2.5-0.5B + LoRA"),
    ("tfidf-logreg", "TF-IDF + regressão logística"),
]
DATASET_COLUMNS = ("review_id", "text", "aspects", "main_issue", "needs_contact")


def fmt(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def results_table() -> str:
    lines = [
        "| Modelo | Macro-F1 aspectos | Macro-F1 `main_issue` | Acurácia `main_issue` "
        "| F1 `needs_contact` |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in RESULT_ROWS:
        m = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))["metrics"]
        lines.append(
            f"| {label} | {fmt(m['aspects_macro_f1'])} | {fmt(m['main_issue_macro_f1'])} "
            f"| {fmt(m['main_issue_accuracy'])} | {fmt(m['needs_contact_f1'])} |"
        )
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_split(records: list[dict], label_source: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            row = {k: r[k] for k in DATASET_COLUMNS}
            row["label_source"] = label_source
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(records)


def latest_gradio_version() -> str | None:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/gradio/json", timeout=10) as r:
            return json.load(r)["info"]["version"]
    except Exception:  # offline or PyPI unavailable: let the Hub pick its default
        return None


MODEL_CARD = """---
base_model: {base_model}
library_name: peft
language:
- pt
license: cc-by-nc-sa-4.0
datasets:
- {user}/{dataset}
pipeline_tag: text-generation
tags:
- lora
- distillation
- customer-reviews
- e-commerce
- portuguese
---

# {model}

Adaptador LoRA para o **{base_model}** que lê uma avaliação de cliente de e-commerce em
português e devolve um **diagnóstico estruturado em JSON**: o que foi elogiado ou criticado,
qual é o problema principal e se o cliente precisa ser contatado.

É o modelo "aluno" de um projeto de **destilação**: um LLM grande ({teacher}) rotulou 2.000
avaliações reais da Olist, e um modelo pequeno foi ajustado para imitá-lo. Código, dados e
análise completa: [{github}]({github}). Demo interativa no Colab (grátis): [abrir]({colab}).

## Exemplo de saída

Entrada: *"Comprei 2 relógios, mas só foi entregue 1"*

```json
{{"aspects": {{"delivery_time": "not_mentioned", "product_quality": "not_mentioned",
  "as_described": "not_mentioned", "order_accuracy": "negative", "packaging": "not_mentioned",
  "customer_service": "not_mentioned"}}, "main_issue": "missing_item", "needs_contact": true}}
```

## Resultados

Teste ouro: 150 avaliações reais, nunca vistas no treino, com rótulos revisados à mão.

{table}

O aluno atinge cerca de 90% da qualidade do professor, rodando localmente e sem custo por
chamada. O número do professor é um teto otimista: o ouro foi revisado a partir dos rótulos
dele. Detalhes, análise de erros e limitações estão no README do GitHub.

## Como usar

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{user}/{model}")
model = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, "{user}/{model}").merge_and_unload().eval()

system = ({system_prompt!r})
messages = [{{"role": "system", "content": system}},
            {{"role": "user", "content": "Veio a cor errada e ninguém me responde."}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Treino

- Dados: 1.800 avaliações de treino e 200 de validação rotuladas por {teacher}
  (dataset [{user}/{dataset}](https://huggingface.co/datasets/{user}/{dataset})).
- LoRA r=16, alpha=32, dropout 0,05, todas as camadas lineares; 3 épocas; lr 2e-4 (cosseno);
  batch 16; fp16; GPU T4 gratuita do Colab (~17 min).
- Os hiperparâmetros foram fixados antes do treino; nada foi ajustado olhando o teste ouro.

## Limitações

- Aprende a imitar o professor, inclusive seus erros sistemáticos.
- Classes raras e vagas (`bad_service`, `other`) continuam fracas (F1 0,29 e 0,38).
- Dados de um marketplace brasileiro de 2016 a 2018; não há garantia de generalização para
  outros domínios, como delivery de comida.

## Licença

Treinado com dados da Olist (CC BY-NC-SA 4.0), por isso o adaptador é distribuído sob a
mesma licença, para uso não comercial. O modelo base ({base_model}) tem licença própria.
"""

DATASET_CARD = """---
language:
- pt
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
tags:
- customer-reviews
- e-commerce
- aspect-based-sentiment-analysis
- distillation
pretty_name: Diagnóstico de avaliações da Olist (PT-BR)
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# Diagnóstico de avaliações da Olist (PT-BR)

Avaliações reais de clientes do e-commerce brasileiro (Olist, 2016 a 2018) com um **diagnóstico
estruturado**: sentimento por aspecto, problema principal e se o cliente precisa ser contatado.
Criado para o projeto [review-distill-ptbr]({github}), que destila um LLM grande num modelo
pequeno ([{user}/{model}](https://huggingface.co/{user}/{model})).

| Split | Avaliações | Origem dos rótulos (`label_source`) |
|---|---:|---|
| `train` | {n_train} | `{teacher_source}`: rótulos gerados pelo LLM professor |
| `validation` | {n_val} | `{teacher_source}`: rótulos gerados pelo LLM professor |
| `test` | {n_test} | `human-reviewed`: teste ouro revisado à mão |

## Campos

- `review_id`, `text`: id e texto da avaliação (título + comentário).
- `aspects`: para `delivery_time`, `product_quality`, `as_described`, `order_accuracy`,
  `packaging` e `customer_service`, um de `positive`, `negative`, `not_mentioned`.
- `main_issue`: `none`, `late_delivery`, `not_delivered`, `wrong_item`, `missing_item`,
  `damaged`, `poor_quality`, `not_as_described`, `bad_service` ou `other`.
- `needs_contact`: o cliente tem um problema sem solução ou pediu uma ação.

## Como foi construído

1. Avaliações com texto, sem duplicatas (36.469 de 99.224), amostradas de forma estratificada
   por nota: 150 para o teste e 2.000 para treino e validação, sem sobreposição.
2. Rótulos gerados por {teacher}, com saída estruturada validada por schema. O prompt passou
   por três versões, validadas em amostras de treino, nunca no teste.
3. O teste foi pré-rotulado pelo mesmo modelo e revisado por uma pessoa (16 de 150 rótulos
   corrigidos). Por partir dos rótulos do professor, a revisão tende a favorecê-lo.

Os rótulos de treino e validação não foram revisados e contêm erros do professor
(concordância de cerca de 89% com a revisão humana no teste).

## Licença e atribuição

Derivado do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
distribuído sob CC BY-NC-SA 4.0; este dataset mantém a mesma licença (uso não comercial,
com atribuição e compartilhamento pela mesma licença).
"""

SPACE_CARD = """---
title: Diagnóstico de avaliações (PT-BR)
emoji: 📦
colorFrom: red
colorTo: yellow
sdk: gradio
{sdk_version_line}app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Diagnóstico de avaliações de clientes com um LLM destilado
models:
- {user}/{model}
- {base_model}
datasets:
- {user}/{dataset}
---

Demo do modelo [{user}/{model}](https://huggingface.co/{user}/{model}).
Código e resultados: [{github}]({github}).
"""


def build_model(staging: Path, adapter_dir: Path, ctx: dict) -> Path:
    from review_distill.inference import STUDENT_SYSTEM_PROMPT

    out = staging / "model"
    shutil.copytree(adapter_dir, out, ignore=shutil.ignore_patterns("checkpoint-*", "README.md"))
    (out / "README.md").write_text(
        MODEL_CARD.format(**ctx, system_prompt=STUDENT_SYSTEM_PROMPT), encoding="utf-8"
    )
    return out


def build_dataset(staging: Path, data_dir: Path, ctx: dict) -> Path:
    out = staging / "dataset"
    teacher_source = "gemini-3.5-flash-lite/prompt-v3"
    gold = read_jsonl(ROOT / "data" / "gold" / "gold_labels.jsonl")
    counts = {
        "n_train": write_split(
            read_jsonl(data_dir / "train.jsonl"), teacher_source, out / "data/train.jsonl"
        ),
        "n_val": write_split(
            read_jsonl(data_dir / "val.jsonl"), teacher_source, out / "data/validation.jsonl"
        ),
        "n_test": write_split(gold, "human-reviewed", out / "data/test.jsonl"),
    }
    card = DATASET_CARD.format(**ctx, **counts, teacher_source=teacher_source)
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def build_space(staging: Path, ctx: dict) -> Path:
    out = staging / "space"
    out.mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", out / "requirements.txt")
    pkg = out / "review_distill"
    pkg.mkdir()
    for name in ("__init__.py", "inference.py", "schema.py"):
        shutil.copy(ROOT / "src" / "review_distill" / name, pkg / name)
    version = latest_gradio_version()
    line = f"sdk_version: {version}\n" if version else ""
    (out / "README.md").write_text(
        SPACE_CARD.format(**ctx, sdk_version_line=line), encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--user", required=True, help="Hugging Face username")
    parser.add_argument("--adapter-dir", type=Path, help="folder with the trained LoRA adapter")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "labeled",
        help="folder with train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["model", "dataset", "space"],
        default=["model", "dataset"],
        help="what to publish (default: model dataset; a Gradio Space needs a PRO account)",
    )
    parser.add_argument("--private", action="store_true", help="create private repos")
    parser.add_argument("--dry-run", action="store_true", help="build folders, do not upload")
    parser.add_argument("--staging", type=Path, help="where to build (default: temp dir)")
    args = parser.parse_args()

    ctx = {
        "user": args.user,
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "space": SPACE_NAME,
        "base_model": BASE_MODEL,
        "teacher": TEACHER,
        "github": GITHUB_URL,
        "table": results_table(),
        "colab": COLAB_DEMO_URL,
    }
    staging = args.staging or Path(tempfile.mkdtemp(prefix="publish_hf_"))
    staging.mkdir(parents=True, exist_ok=True)

    built: list[tuple[str, str, Path]] = []
    if "dataset" in args.only:
        built.append(
            ("dataset", f"{args.user}/{DATASET_NAME}", build_dataset(staging, args.data_dir, ctx))
        )
    if "model" in args.only:
        if args.adapter_dir is None:
            parser.error("--adapter-dir is required to publish the model")
        built.append(
            ("model", f"{args.user}/{MODEL_NAME}", build_model(staging, args.adapter_dir, ctx))
        )
    if "space" in args.only:
        built.append(("space", f"{args.user}/{SPACE_NAME}", build_space(staging, ctx)))

    for kind, repo_id, folder in built:
        print(f"[{kind}] {repo_id} <- {folder}")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    for kind, repo_id, folder in built:
        repo_type = None if kind == "model" else kind
        kwargs = {"space_sdk": "gradio"} if kind == "space" else {}
        api.create_repo(repo_id, repo_type=repo_type, private=args.private, exist_ok=True, **kwargs)
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=folder,
            commit_message="Publish from scripts/publish_hf.py",
        )
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[kind]
        print(f"published: https://huggingface.co/{prefix}{repo_id}")


if __name__ == "__main__":
    main()
