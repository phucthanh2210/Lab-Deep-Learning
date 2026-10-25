"""Pretrained inference and a freshly initialized classification head for tuning."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

def load_model(name, revision=None, finetuning=False):
    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    model_kwargs = dict(num_labels=2, id2label=LABELS,
                        label2id={v: k for k, v in LABELS.items()}) if finetuning else {}
    model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs, **model_kwargs)
    return model, tokenizer

def manual_predict(model, tokenizer, texts, max_length=128):
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = logits.softmax(-1).cpu().tolist()
    rows = []
    for text, scores in zip(texts, probs):
        idx = max(range(len(scores)), key=scores.__getitem__)
        rows.append(dict(text=text, label=model.config.id2label[idx],
                         score=scores[idx], probabilities=scores))
    return rows

def compare_inference(model, tokenizer, texts, max_length=128):
    device = next(model.parameters()).device
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    high = pipe(texts, truncation=True, max_length=max_length)
    low = manual_predict(model, tokenizer, texts, max_length)
    for a, b in zip(high, low):
        if a["label"] != b["label"] or abs(a["score"] - b["score"]) > 1e-5:
            raise AssertionError(f"Pipeline/manual mismatch: {a}, {b}")
    return high, low
