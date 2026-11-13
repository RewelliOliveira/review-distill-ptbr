# review-distill-ptbr

Destilação de um LLM grande (Gemini) em um modelo pequeno (~0,5–1,5B parâmetros) para extrair **diagnósticos estruturados** de avaliações de clientes de e-commerce em português — com qualidade próxima à do modelo grande e uma fração do custo e da latência.

> 🚧 Em construção. Links para modelo, dataset, demo e notebook serão adicionados na publicação.

## O problema

Plataformas de e-commerce e delivery recebem milhares de avaliações por dia. A nota (1–5 estrelas) diz *se* o cliente ficou insatisfeito, mas não *por quê*, nem *se alguém precisa agir*. Ler tudo à mão não escala, e chamar um LLM grande para cada avaliação custa caro e é lento.

A pergunta deste projeto: **um modelo pequeno, ajustado com rótulos gerados por um LLM grande, consegue fazer esse diagnóstico com qualidade parecida, por uma fração do custo?**

Para cada avaliação, o modelo produz um JSON:

```json
{
  "aspects": {
    "delivery_time": "negative",
    "product_quality": "not_mentioned",
    "as_described": "not_mentioned",
    "order_accuracy": "negative",
    "packaging": "not_mentioned",
    "customer_service": "negative"
  },
  "main_issue": "missing_item",
  "needs_contact": true
}
```

| Campo | Valores |
|---|---|
| `aspects` (6 aspectos) | `positive` · `negative` · `not_mentioned` para prazo de entrega, qualidade do produto, produto igual ao anunciado, pedido correto (item errado/faltando), embalagem e atendimento |
| `main_issue` | `none` · `late_delivery` · `not_delivered` · `wrong_item` · `missing_item` · `damaged` · `poor_quality` · `not_as_described` · `bad_service` · `other` |
| `needs_contact` | o cliente tem um problema não resolvido e precisa ser contatado? |

A definição oficial está em [`src/review_distill/schema.py`](src/review_distill/schema.py).

## Abordagem

1. **Dados** — avaliações reais e públicas da [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~99 mil avaliações, 2016–2018).
2. **Professor** — o Gemini rotula alguns milhares de avaliações com saída estruturada (JSON restrito ao schema), em lotes, com cada item validado por pydantic.
3. **Teste ouro** — 150 avaliações (30 por nota) revisadas à mão, usadas só na avaliação final (nunca em treino nem em ajuste de prompt).
4. **Baselines** — TF-IDF + regressão logística, BERTimbau e Gemini zero-shot.
5. **Aluno** — modelo pequeno (Qwen2.5-0.5B/1.5B-Instruct) ajustado com LoRA nos rótulos do professor.
6. **Avaliação** — F1 por aspecto, % de JSON válido, latência e custo por 1.000 avaliações, todos no mesmo teste ouro e com o mesmo script.

## Dados

Funil de limpeza do arquivo `olist_order_reviews_dataset.csv` (detalhes em [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)):

| Etapa | Avaliações | % do total |
|---|---:|---:|
| Linhas no arquivo | 99.224 | 100,0 |
| Com algum texto (título ou comentário) | 42.687 | 43,0 |
| Sem `review_id` repetido | 42.370 | 42,7 |
| Sem texto duplicado (ignorando maiúsculas) | **36.469** | **36,8** |

Descartes e motivos:

- **Avaliações sem texto (57%)** — só têm nota; não há o que diagnosticar.
- **`review_id` repetido** — a Olist reutiliza o mesmo id de avaliação em pedidos diferentes; mantemos a primeira ocorrência.
- **Textos duplicados** — principalmente textos curtos ("Muito bom", "Recomendo"); removidos antes da divisão treino/teste para evitar vazamento.

Os textos são curtos (mediana de 10 palavras, máximo de 48) e as avaliações negativas são mais longas que as positivas. Entre as avaliações com texto, 48% têm nota 5 e 23% nota 1.

## Resultados

Todos os modelos são avaliados no mesmo teste ouro (150 avaliações revisadas), com o mesmo script ([`scripts/evaluate.py`](scripts/evaluate.py)) e as mesmas métricas ([`src/review_distill/metrics.py`](src/review_distill/metrics.py)). Os números completos, com F1 por aspecto e por classe e a configuração de cada modelo, estão em [`results/`](results/).

| Modelo | Macro-F1 aspectos | Macro-F1 `main_issue` | Acurácia `main_issue` | F1 `needs_contact` | JSON válido | Latência (ms/aval.) | Custo / 1.000 aval. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash-Lite, prompt v3 (professor)¹ | 0,970 | 0,921 | 0,960 | 0,957 | 100% | 372² | ~121 mil tokens de entrada + ~140 mil de saída |
| TF-IDF + regressão logística³ | 0,690 | 0,632 | 0,740 | 0,904 | 100%⁴ | 2,9 | ≈ 0 (CPU local) |
| BERTimbau | – | – | – | – | – | – | – |
| Qwen2.5-0.5B + LoRA (aluno) | – | – | – | – | – | – | – |

- **Macro-F1 aspectos:** para cada um dos 6 aspectos, F1 macro sobre `positive`, `negative` e `not_mentioned`; depois, a média dos 6. **Macro-F1 `main_issue`:** média do F1 das 10 classes. **F1 `needs_contact`:** F1 da classe positiva (precisa de contato). Uma saída inválida conta como erro em todos os campos.
- ¹ **O número do professor é um teto otimista.** O teste ouro foi pré-revisado com auxílio de LLM, partindo dos rótulos do próprio Gemini, e a decisão final sobre cada rótulo foi humana: 16 dos 150 rótulos foram corrigidos. Partir das respostas do professor tende a ancorar o revisor nelas, o que favorece o Gemini nesta comparação. Os modelos treinados com os rótulos do Gemini (o aluno e o TF-IDF) não têm essa vantagem.
- ² Tempo de resposta da API dividido pelas 20 avaliações de cada requisição (lote), com rede e sem contar as novas tentativas após erros 503.
- ³ Treinado nas 1.800 avaliações de treino rotuladas pelo Gemini, com hiperparâmetros fixos (sem ajuste). Na validação (200 avaliações), o macro-F1 foi 0,755 nos aspectos e 0,738 em `main_issue`, e o F1 de `needs_contact` foi 0,866. Latência de uma avaliação por vez, em CPU.
- ⁴ Válido por construção: um classificador independente por campo sempre produz valores do schema.

O TF-IDF vai bem em `needs_contact` e em problemas com vocabulário marcado (`not_delivered`: F1 0,83; `missing_item`: 0,82), mas erra mais nas classes raras e vagas (`bad_service`: 0,33; `other`: 0,38) e no aspecto `as_described` (0,51), que depende de entender a comparação com o anúncio.

## Como reproduzir

```bash
uv sync                                        # dependências (CPU)
cp .env.example .env                           # preencher GOOGLE_API_KEY
uv run python scripts/download_data.py         # baixa os dados da Olist para data/raw/
uv run python scripts/make_samples.py          # separa o teste ouro (150) e o pool (2.000)
uv run python scripts/label.py --sample pool   # rotula o pool com o Gemini (retomável)
uv run python scripts/label.py --sample gold   # pré-rotula o ouro para revisão manual
uv run python scripts/review_gold.py           # revisão manual do ouro
uv run python scripts/make_splits.py           # treino (90%) e validação (10%), estratificados por main_issue
uv run python scripts/evaluate.py --model gemini-teacher   # avalia o professor no ouro
uv run python scripts/evaluate.py --model tfidf-logreg     # treina e avalia o baseline TF-IDF
uv run pytest
```

### Fine-tuning no Colab

O fine-tuning roda em GPU gratuita (T4) no notebook [`notebooks/02_finetune.ipynb`](notebooks/02_finetune.ipynb). Ele clona o repositório, instala o extra `train`, ajusta o Qwen2.5-0.5B-Instruct com LoRA, avalia na validação, gera as previsões no ouro e as avalia com o mesmo `scripts/evaluate.py`.

1. No Colab, abra o notebook (*Arquivo → Abrir notebook → GitHub*) e escolha *Ambiente de execução → Alterar o tipo → GPU T4*.
2. Adicione o token do Hugging Face como secret: ícone 🔑 na barra lateral → `HF_TOKEN` → ative o acesso do notebook. O token só é obrigatório para publicar o adaptador no Hub (`PUSH_TO_HUB = True`).
3. Forneça `train.jsonl` e `val.jsonl`. Eles não estão no git: gere-os com `scripts/make_splits.py` (ficam em `data/labeled/`) e escolha na célula de configuração:
   - `DATA_SOURCE = "upload"`: o notebook abre um seletor de arquivos e você envia os dois. É preciso repetir a cada sessão, e no fim os resultados são baixados num `.zip`.
   - `DATA_SOURCE = "drive"`: copie os dois arquivos uma vez para `Meu Drive/review-distill/`. O notebook monta o Drive, lê os arquivos de lá e salva ali o adaptador e as previsões, então nada se perde se a sessão cair.
4. Execute tudo. Para versionar o resultado, copie `results/<nome>.json` e `results/predictions/<nome>*` para o repositório.

### Com Docker (sem instalar Python)

Se você não quer instalar Python e uv, ou se o Windows bloqueia bibliotecas compiladas (por exemplo, pelo Smart App Control), dá para rodar tudo num container Linux. Basta ter o [Docker Desktop](https://www.docker.com/products/docker-desktop/) aberto.

Na pasta do projeto, no **PowerShell**:

```powershell
docker run -it --rm -v "${PWD}:/work" -v review-venv:/venv -e UV_PROJECT_ENVIRONMENT=/venv -w /work ghcr.io/astral-sh/uv:bookworm-slim bash -c "uv sync && uv run pytest"
```

No **Linux, macOS ou WSL**:

```bash
docker run -it --rm -v "$(pwd):/work" -v review-venv:/venv -e UV_PROJECT_ENVIRONMENT=/venv -w /work ghcr.io/astral-sh/uv:bookworm-slim bash -c "uv sync && uv run pytest"
```

Para rodar outro comando, troque `uv run pytest` no final. Por exemplo, `uv run python scripts/review_gold.py` para revisar o teste ouro.

- `-v ...:/work` compartilha a pasta do projeto com o container. Os arquivos gerados (rótulos, resultados) ficam salvos no seu computador.
- `-v review-venv:/venv` guarda as dependências num volume do Docker. A primeira execução leva alguns minutos, e as seguintes já começam com tudo instalado.
- `-it` permite interagir com os scripts pelo terminal.
- O `.env` com as chaves é lido da pasta do projeto, como na execução local.

## Limitações

- **Rótulos do professor não são verdade absoluta.** O aluno aprende a imitar o Gemini, inclusive seus erros; o teste ouro revisado à mão mede o quanto isso pesa.
- **Teste ouro pequeno (150 avaliações).** Classes raras têm poucos exemplos (ex.: 2 de `bad_service`, 5 de `not_as_described`), e as métricas delas têm intervalos de confiança largos.
- **Ouro ancorado no professor.** A revisão partiu dos rótulos do Gemini, o que tende a favorecer o professor na comparação (ver [Resultados](#resultados)).
- **Revisão por uma só pessoa.** Não há medida de concordância entre anotadores.
- **Domínio e época específicos.** Dados de um marketplace brasileiro de 2016–2018; os nomes de lojas foram anonimizados pela Olist. Não há garantia de que o modelo generalize para delivery de comida ou outras plataformas.
- **Só texto.** A nota em estrelas e os dados do pedido (ex.: datas de entrega) não são usados como entrada.
- **Textos curtos e ambíguos.** Cerca de 10% das avaliações têm até 3 palavras ("Bom", "Não recebi"), o que limita o que qualquer modelo consegue inferir.

*Esta seção será complementada com a análise de erros do modelo final.*

## Licença

Código sob licença MIT. Os dados da Olist são distribuídos sob CC BY-NC-SA 4.0.
