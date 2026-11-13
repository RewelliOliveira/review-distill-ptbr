# Teacher prompt changelog

History of the labeling prompt (`src/review_distill/prompts.py`, `PROMPT_VERSION`) used by the
teacher model to label the train/val pool and pre-label the gold set.

**The gold test set (`data/gold/gold_reviews.csv`, 150 reviews) was never used to design, tune or
validate any prompt version.** All iterations below used only reviews from the train/val pool
(`data/labeled/pool_reviews.csv`), taken in pool order (the pool is stored shuffled with seed 42,
so slices are roughly score-balanced and were not hand-picked). The gold set was labeled once,
with the final prompt, after the prompt was frozen.

## Setup

- Teacher: `gemini-3.5-flash-lite`, thinking level `low`, temperature 0, seed 42, batches of 20.
  Re-running v2 on the same 60 reviews produced identical labels (60/60), so differences
  between versions below come from the prompt, not sampling noise.
- Reference: `gemini-3.8-flash` with prompt v1. It was the first teacher, dropped because one
  batch of 20 took 538 s versus 13.5 s for flash-lite (1 attempt, no 429, see below). It is used
  only as a second opinion; agreement with it is not accuracy.
- Pilot slices: rows 1-60 and 61-120 of the pool.
- Agreement is the number of reviews (out of 60) with the same value in each field;
  "all" means every field equal. Column abbreviations: prazo = delivery_time,
  qual = product_quality, anunc = as_described, pedido = order_accuracy, emb = packaging,
  atend = customer_service.
- Pilot labels live in `data/labeled/pilot/` (not versioned, like all bulk labels).

## v1 (commit 0d5dc59)

Initial prompt: aspect definitions, main_issue severity order, needs_contact definition.

- Validated on: rows 1-60 (pilot), reviewed by the project author (15 examples).
- Model choice: teacher switched from `gemini-3.8-flash` to `gemini-3.5-flash-lite` (latency;
  see setup). On the same 20 reviews, flash-lite matched every field on 17/20.

| Comparison (rows 1-60) | prazo | qual | anunc | pedido | emb | atend | main_issue | needs_contact | all |
|---|---|---|---|---|---|---|---|---|---|
| lite v1 × 3.8-flash v1 | 58 | 56 | 58 | 59 | 60 | 59 | 57 | 54 | 45 |

Issues found in the pilot review (ambiguities in the prompt, not model noise):
- No rule on whether a product that differs from the listing needs contact when the customer asks
  for nothing ("as cores nao sao da propaganda"; "imagem com cabo inox, recebido com cabo de
  madeira"): the two models disagreed.
- A different variant was marked both as_described and order_accuracy negative
  ("Pedi a saia branca me mandaram creme").
- Transport damage also marked product_quality negative ("caixa machucada… vidro quebrado").

## v2 (commit 58f67a8)

Changes (approved by the project author), each with one short synthetic example in the prompt:
1. Wrong variant (color, model, size) is order_accuracy only, not as_described.
2. Transport/handling damage is packaging + damaged, not product_quality.
3. not_as_described always needs contact.

- Validated on: rows 1-60.

| Comparison (rows 1-60) | prazo | qual | anunc | pedido | emb | atend | main_issue | needs_contact | all |
|---|---|---|---|---|---|---|---|---|---|
| lite v2 × lite v1 | 60 | 56 | 54 | 60 | 60 | 59 | 56 | 57 | 44 |
| lite v2 × 3.8-flash v1 | 58 | 56 | 54 | 59 | 60 | 60 | 55 | 51 | 38 |
| lite v2 × lite v2 (re-run) | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 60 |

The three target cases changed as intended, but v2 introduced clear regressions:
- "Comprei duas cortinas, porém só chegou uma" became wrong_item (should be missing_item); the
  wrong-variant example seems to have pulled missing items into wrong_item.
- "Não recebi ainda. MAS está ainda no prazo de entrega" became not_delivered (with no contact).
- "Razoável" became product_quality positive.
- "Recebi apenas um de dois… abri chamado para… estornar… boneco falso" lost needs_contact and
  as_described negative.
- needs_contact flipped on 3 cases no rule covered; agreement with the reference dropped to 51/60.

## v3 (commit a4b3759) — final

Changes, each with a short synthetic example:
1. Explicit split between missing_item (fewer items arrived), wrong_item (a different variant
   arrived; order_accuracy only) and not_as_described (item differs from the listing or is
   counterfeit; as_described negative).
2. not_delivered only when the delivery date has passed; "not yet, but within the deadline"
   is none with no contact.
3. needs_contact written as a positive list (late non-delivery; wrong/missing/damaged/
   not-as-described item; explicit request for refund, exchange, cancellation or reply) and
   false for vague negative opinion and for poor quality without a request.
4. Generic or lukewarm words ("bom", "razoável", "ok") do not set any aspect; scratched counts
   as transport damage; authenticity belongs to as_described.

- Validated on: rows 1-60 (regression check) and rows 61-120 (unseen by any earlier version).
- Stopping rule set in advance by the project author: approve if at most 3 clear errors on the
  60 unseen reviews; otherwise one more version at most.

| Comparison | prazo | qual | anunc | pedido | emb | atend | main_issue | needs_contact | all |
|---|---|---|---|---|---|---|---|---|---|
| rows 1-60: lite v3 × lite v2 | 58 | 54 | 56 | 60 | 60 | 59 | 54 | 54 | 39 |
| rows 1-60: lite v3 × 3.8-flash v1 | 58 | 55 | 56 | 59 | 60 | 59 | 53 | 55 | 42 |
| rows 61-120: lite v3 × lite v2 | 56 | 59 | 59 | 60 | 60 | 58 | 57 | 53 | 45 |
| rows 61-80: lite v3 × 3.8-flash v1 (n=20, partial*) | 20 | 20 | 16 | 20 | 20 | 19 | 19 | 17 | 12 |
| rows 61-80: lite v2 × 3.8-flash v1 (n=20, partial*) | 18 | 20 | 16 | 20 | 20 | 19 | 18 | 18 | 11 |

\* The reference run on rows 61-120 was interrupted by repeated 503 (overloaded) errors from
`gemini-3.8-flash`; only the first 20 reviews were labeled so far. To be completed.

On rows 1-60, v3 fixed all four v2 regressions listed above and kept the three v2 fixes.

Clear errors on rows 61-120 (3, at the approval limit):
- "Não condiz com o original. O cartucho que seria de alto rendimento… não durou mais que 100
  impressões" → not_as_described with needs_contact false (the prompt says true).
- "Não funciona, mais uma vez meu produto chegou com defeito!" → poor_quality with
  needs_contact false (defective on arrival is an unresolved problem).
- "Comprei 3 luminárias e recebi apenas 2… Abri reclamação pelo site… aguardo retorno" →
  customer_service positive (nothing positive about service is said).

Debatable, not counted: paid for home delivery but had to pick up at the post office
(other + contact); "acho um descaso com o consumidor" (bad_service + contact).

Known weakness carried into the dataset: flash-lite does not always apply "not_as_described ⇒
needs_contact" (also seen on rows 1-60: "Achei o produto muito pequeno, e o material fraco").
The hand-reviewed gold set will measure how often this happens.

**Decision:** v3 approved under the stopping rule (3 clear errors on 60 unseen reviews). No v4.

## Full labeling run (v3)

Teacher `gemini-3.5-flash-lite`, prompt v3, thinking `low`, seed 42, batches of 20, client-side
limit of 4 requests/min. The run was interrupted and resumed (cache by review id), so the
labels come from several sessions on 2026-11-11, all with the same model and prompt.

| Sample | Reviews | Valid | Invalid | Requests | Failed requests (retried later) | 503 retries | 429 |
|---|---|---|---|---|---|---|---|
| pool (train/val) | 2,000 | 2,000 | 0 | 101 | 1 | 43 | 0 |
| gold (pre-label for hand review) | 150 | 150 | 0 | 10 | 2 | 20 | 0 |

Time: 1,718 s summed over all requests (API time including retries); the final resumed session
(1,760 pool + 70 gold reviews) took 1,477 s wall clock, dominated by the 4 req/min limit.

Label distribution:

| main_issue | pool | % | gold | % |
|---|---|---|---|---|
| none | 784 | 39.2 | 56 | 37.3 |
| missing_item | 218 | 10.9 | 10 | 6.7 |
| not_delivered | 205 | 10.2 | 22 | 14.7 |
| poor_quality | 185 | 9.2 | 15 | 10.0 |
| late_delivery | 146 | 7.3 | 17 | 11.3 |
| not_as_described | 135 | 6.8 | 4 | 2.7 |
| wrong_item | 117 | 5.8 | 8 | 5.3 |
| damaged | 73 | 3.6 | 7 | 4.7 |
| other | 69 | 3.5 | 8 | 5.3 |
| bad_service | 68 | 3.4 | 3 | 2.0 |

| needs_contact | pool | % | gold | % |
|---|---|---|---|---|
| true | 798 | 39.9 | 57 | 38.0 |
| false | 1,202 | 60.1 | 93 | 62.0 |

The gold labels above are the teacher's pre-labels only (`data/gold/gemini_labels.jsonl`); the
reference labels are the hand-reviewed ones in `data/gold/gold_labels.jsonl`.
