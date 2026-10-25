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


def verify_saved_model(reference_model, model_path, texts, max_length=128):
    """Check exact weights, then inference in a common CPU FP32 execution context.

    Trainer/Accelerate can wrap the live forward method with CUDA autocast.
    Compare clean instances instead of comparing that wrapper to a CPU reload.
    """
    reloaded, tokenizer = load_model(str(model_path))
    expected = reference_model.state_dict()
    actual = reloaded.state_dict()
    if expected.keys() != actual.keys():
        raise AssertionError("Checkpoint parameter names differ")
    for name in actual:
        if not torch.equal(actual[name].cpu(), expected[name].cpu()):
            raise AssertionError(f"Checkpoint weight differs: {name}")
    clean_reference = type(reloaded)(reloaded.config)
    clean_reference.load_state_dict(expected)
    clean_reference.float().cpu()
    reloaded.float().cpu()
    original = manual_predict(clean_reference, tokenizer, texts, max_length)
    checks = manual_predict(reloaded, tokenizer, texts, max_length)
    for before, after in zip(original, checks):
        if before["label"] != after["label"] or not torch.allclose(
            torch.tensor(before["probabilities"]), torch.tensor(after["probabilities"]),
            atol=1e-6, rtol=1e-5):
            raise AssertionError("Checkpoint reload predictions differ in matched CPU FP32 context")
    return {"weights_exact_match": True, "execution": "CPU FP32, clean forward methods",
            "predictions": checks}
