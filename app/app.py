"""Gradio demo: diagnose a Portuguese customer review with the distilled student model.

Runs on Colab (notebooks/03_demo.ipynb, GPU or CPU), locally (`python app/app.py`) or on a
Hugging Face Space. The LoRA adapter is downloaded from the Hub and merged into the base
model at startup; the prompt format and the output parser are the ones used in training
and evaluation (review_distill.inference). Set SHARE=1 to get a public gradio.live link.
"""

import json
import os
import time

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from review_distill.inference import parse_diagnosis, prompt_messages

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "Rewelli/review-distill-qwen2.5-1.5b-lora")
MAX_NEW_TOKENS = 128

ASPECT_LABELS = {
    "delivery_time": "Prazo de entrega",
    "product_quality": "Qualidade do produto",
    "as_described": "Igual ao anúncio",
    "order_accuracy": "Pedido correto (item certo e completo)",
    "packaging": "Embalagem",
    "customer_service": "Atendimento",
}
SENTIMENT_LABELS = {
    "positive": "🟢 positivo",
    "negative": "🔴 negativo",
    "not_mentioned": "⚪ não mencionado",
}
ISSUE_LABELS = {
    "none": "Nenhum problema",
    "late_delivery": "Entrega atrasada",
    "not_delivered": "Não entregue",
    "wrong_item": "Item errado",
    "missing_item": "Item faltando",
    "damaged": "Produto danificado",
    "poor_quality": "Baixa qualidade / defeito",
    "not_as_described": "Diferente do anúncio",
    "bad_service": "Atendimento ruim",
    "other": "Outro",
}

EXAMPLES = [
    "Comprei 3 camisetas e só chegaram 2. Aguardo retorno da loja.",
    "Chegou antes do prazo e o produto é excelente, recomendo!",
    "O produto veio quebrado e ninguém responde meus e-mails.",
    "A cor é bem diferente da foto do anúncio, me arrependi.",
    "Já passou do prazo de entrega e até agora não recebi nada.",
    "Produto bom, mas a caixa chegou toda amassada.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_LABEL = "numa GPU" if DEVICE == "cuda" else "numa CPU"
torch.set_num_threads(os.cpu_count() or 2)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(_base, ADAPTER_REPO).merge_and_unload()
if DEVICE == "cuda":
    model = model.half()
model = model.to(DEVICE).eval()


def generate(text: str) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        prompt_messages(text), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    seconds = time.perf_counter() - start
    raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return raw, seconds


def diagnose(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "Escreva uma avaliação para analisar.", ""
    raw, seconds = generate(text[:1000])
    diagnosis = parse_diagnosis(raw)
    if diagnosis is None:
        return f"⚠️ O modelo não gerou um JSON válido.\n\nSaída bruta: `{raw}`", raw
    d = diagnosis.model_dump(mode="json")
    contact = (
        "🚨 **Sim**: o cliente tem um problema sem solução ou pediu uma ação."
        if d["needs_contact"]
        else "✅ **Não**"
    )
    rows = "\n".join(
        f"| {ASPECT_LABELS[k]} | {SENTIMENT_LABELS[v]} |" for k, v in d["aspects"].items()
    )
    summary = (
        f"### Problema principal: {ISSUE_LABELS[d['main_issue']]}\n\n"
        f"**Precisa contatar o cliente?** {contact}\n\n"
        f"| Aspecto | Avaliação |\n|---|---|\n{rows}\n\n"
        f"<sub>Gerado em {seconds:.1f} s {DEVICE_LABEL}.</sub>"
    )
    return summary, json.dumps(d, ensure_ascii=False, indent=2)


DESCRIPTION = """
Modelo **Qwen2.5-1.5B** ajustado com LoRA a partir de rótulos do Gemini (destilação), para
diagnosticar avaliações de clientes de e-commerce em português: o que foi elogiado ou
criticado, qual o problema principal e se alguém precisa contatar o cliente.

No teste com 150 avaliações reais revisadas à mão, ele atinge cerca de 90% da qualidade do
modelo professor, rodando localmente e sem custo por chamada.
[Código e resultados no GitHub](https://github.com/RewelliOliveira/review-distill-ptbr).

Numa GPU cada análise leva menos de um segundo; numa CPU, alguns segundos.
"""

with gr.Blocks() as demo:
    gr.Markdown("# Diagnóstico de avaliações de clientes")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            review = gr.Textbox(
                label="Avaliação do cliente",
                placeholder="Ex.: Comprei dois produtos e só chegou um. Aguardo retorno.",
                lines=4,
            )
            button = gr.Button("Analisar", variant="primary")
            gr.Examples(EXAMPLES, inputs=review)
        with gr.Column():
            result = gr.Markdown()
            with gr.Accordion("JSON gerado pelo modelo", open=False):
                raw_json = gr.Code(language="json")
    button.click(diagnose, inputs=review, outputs=[result, raw_json])
    review.submit(diagnose, inputs=review, outputs=[result, raw_json])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(share=os.environ.get("SHARE") == "1")
