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
3. **Teste ouro** — ~200 avaliações revisadas à mão, usadas só na avaliação final (nunca em treino nem em ajuste de prompt).
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

*Ainda não há resultados.* Todos os modelos serão avaliados no mesmo teste ouro.

| Modelo | Macro-F1 aspectos | Acurácia `main_issue` | F1 `needs_contact` | JSON válido | Latência (ms/aval.) | Custo / 1.000 aval. |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + regressão logística | – | – | – | – | – | – |
| BERTimbau | – | – | – | – | – | – |
| Gemini zero-shot (professor) | – | – | – | – | – | – |
| Qwen2.5 + LoRA (aluno) | – | – | – | – | – | – |

## Como reproduzir

```bash
uv sync                                        # dependências (CPU)
cp .env.example .env                           # preencher GOOGLE_API_KEY
uv run python scripts/download_data.py         # baixa os dados da Olist para data/raw/
uv run python scripts/make_samples.py          # separa o teste ouro (150) e o pool (2.000)
uv run python scripts/label.py --sample pool   # rotula o pool com o Gemini (retomável)
uv run python scripts/label.py --sample gold   # pré-rotula o ouro para revisão manual
uv run pytest
```

O fine-tuning roda em GPU gratuita (Colab/Kaggle, T4) com `uv sync --extra train`.

## Limitações

- **Rótulos do professor não são verdade absoluta.** O aluno aprende a imitar o Gemini, inclusive seus erros; o teste ouro revisado à mão mede o quanto isso pesa.
- **Teste ouro pequeno (~200 avaliações).** Classes raras (ex.: `wrong_item`, `packaging`) terão poucos exemplos, e as métricas delas terão intervalos de confiança largos.
- **Revisão por uma só pessoa.** Não há medida de concordância entre anotadores.
- **Domínio e época específicos.** Dados de um marketplace brasileiro de 2016–2018; os nomes de lojas foram anonimizados pela Olist. Não há garantia de que o modelo generalize para delivery de comida ou outras plataformas.
- **Só texto.** A nota em estrelas e os dados do pedido (ex.: datas de entrega) não são usados como entrada.
- **Textos curtos e ambíguos.** Cerca de 10% das avaliações têm até 3 palavras ("Bom", "Não recebi"), o que limita o que qualquer modelo consegue inferir.

*Esta seção será complementada com a análise de erros do modelo final.*

## Licença

Código sob licença MIT. Os dados da Olist são distribuídos sob CC BY-NC-SA 4.0.
