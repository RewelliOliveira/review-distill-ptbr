# review-distill-ptbr

Destilação de um LLM grande (Gemini) em um modelo pequeno (~0,5–1,5B) para extrair diagnósticos estruturados de avaliações de clientes de e-commerce/delivery em português. Objetivo: qualidade próxima ao modelo grande com uma fração do custo e da latência.

Projeto de portfólio para a trilha AI Engineer (iFood GEN AI). Entregáveis: código no GitHub, modelo e dataset no Hugging Face, demo em Space (Gradio), notebook no Kaggle e um post técnico.

## Pipeline

1. **Dados** — avaliações reais do dataset público da Olist (Kaggle, `olistbr/brazilian-ecommerce`), arquivo `olist_order_reviews_dataset.csv`
2. **Rotulagem (professor)** — Gemini rotula ~3–5k avaliações com saída estruturada validada pelo schema
3. **Teste ouro** — ~200 avaliações revisadas à mão; nunca usadas em treino nem em ajuste de prompt
4. **Baselines** — TF-IDF + regressão logística; BERTimbau; Gemini zero-shot
5. **Fine-tuning (aluno)** — modelo pequeno com LoRA (Qwen2.5-0.5B/1.5B-Instruct ou similar)
6. **Avaliação** — F1 por aspecto, % JSON válido, latência, custo por 1.000 avaliações, análise de erros
7. **Publicação** — Hugging Face (modelo + dataset + Space) e Kaggle

## Schema de saída (`src/review_distill/schema.py` é a fonte da verdade)

- `aspects`: para cada aspecto → `positive` | `negative` | `not_mentioned`
  - `delivery_time` (prazo/atraso) · `product_quality` · `as_described` (produto igual ao anunciado) · `order_accuracy` (item errado/faltando) · `packaging` · `customer_service` (vendedor/atendimento)
- `main_issue`: `none` | `late_delivery` | `not_delivered` | `wrong_item` | `missing_item` | `damaged` | `poor_quality` | `not_as_described` | `bad_service` | `other`
- `needs_contact`: bool — o cliente precisa ser contatado

Qualquer mudança no schema exige re-rotular os dados afetados e atualizar os testes.

## Stack

- Python 3.11+, gerenciado com **uv** (`pyproject.toml`)
- pandas, scikit-learn (baselines e métricas), pydantic (schema)
- `google-genai` para rotular com Gemini (saída estruturada via schema)
- transformers, datasets, peft, trl, accelerate (fine-tuning); bitsandbytes só no ambiente com GPU
- gradio (demo), huggingface_hub (publicação)
- pytest e ruff
- Treino roda em GPU gratuita (Google Colab ou Kaggle, T4); o resto roda localmente em CPU

## Comandos

- `uv sync` — instalar dependências
- `uv run python scripts/download_data.py` — baixar dados da Olist (requer Kaggle API token)
- `uv run python scripts/label.py --n 500` — rotular com Gemini (retomável)
- `uv run python scripts/make_splits.py` — gerar train/val/test
- `uv run python scripts/evaluate.py --model <nome>` — avaliar no teste ouro
- `uv run pytest` / `uv run ruff check .` / `uv run ruff format .`

## Estrutura

```
src/review_distill/   # pacote: schema, prompts, labeling, metrics, inference
scripts/              # pontos de entrada da linha de comando
notebooks/            # EDA e fine-tuning (o de treino roda no Colab/Kaggle)
data/raw/             # dados originais (não versionados)
data/labeled/         # saídas do Gemini em JSONL (não versionadas; publicadas no HF)
data/gold/            # teste ouro revisado à mão (versionado)
results/              # métricas (JSON/CSV) e gráficos — versionados
app/                  # demo Gradio para o Space
tests/
```

## Regras de ML (não violar)

- **Sem vazamento:** o teste ouro nunca entra em treino, validação ou ajuste de prompt. Deduplicar textos antes de dividir.
- **Reprodutibilidade:** seed fixa em tudo; registrar versão do modelo, prompt e hiperparâmetros em `results/` junto com as métricas.
- **Rotulagem robusta:** lotes de avaliações por requisição (padrão 20, para caber no limite diário do plano gratuito), cache por id de avaliação (JSONL append), retomável após falha, retry com backoff em erro 429, validar cada item com pydantic e registrar os inválidos à parte.
- **Comparação justa:** todos os modelos avaliados no mesmo teste ouro, com o mesmo script de métricas.
- **Honestidade nos resultados:** reportar onde o modelo pequeno perde; análise de erros com exemplos reais no README.
- Avaliações vazias ou só com nota (sem texto) são descartadas e o motivo documentado.

## Convenções

- **Idioma:** README, post e textos da demo em português. Todo o resto em inglês: código, variáveis, arquivos, comentários, commits.
- **Nomes:** módulos e arquivos em `snake_case`; classes em `PascalCase`; constantes em `UPPER_SNAKE_CASE`
- Type hints em funções públicas; configuração via `.env` (`GOOGLE_API_KEY`, `HF_TOKEN`) — manter `.env.example` atualizado
- Notebooks finos: a lógica fica em `src/`, o notebook só orquestra e mostra resultados

## Commits

Conventional Commits, em inglês, imperativo, minúsculo, sem ponto final, até ~72 caracteres. Um commit por passo lógico.

- **Tipos:** `feat` · `fix` · `test` · `refactor` · `docs` · `chore` · `exp` (experimento: nova rodada de treino/prompt, com resultado no corpo)
- **Escopos:** `data` · `labeling` · `baseline` · `train` · `eval` · `app` · `publish`

```
feat(labeling): add resumable gemini labeler with schema validation
exp(train): lora r=16 on qwen2.5-1.5b, macro-f1 0.81 on gold set
docs: add error analysis to readme
```

## Não fazer

- Não versionar dados brutos, rotulados em massa, checkpoints nem pesos (`.gitignore`)
- Não ajustar prompt ou hiperparâmetros olhando o teste ouro — usar a validação
- Não adicionar dependências sem necessidade clara
- Não colocar chaves em código ou notebook
