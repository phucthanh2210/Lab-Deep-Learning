"""Read-only audit of actual run artifacts; never loads models or downloads data."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from .utils.runs import ROOT, hash_file
from .training.metrics import compute_metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_run(kind, root=ROOT):
    root = Path(root)
    state = read_json(root / "config/workflow.json")
    run_id = state["active_runs"].get(kind)
    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"No valid selected {kind} run in config/workflow.json")
    path = root / "outputs/runs" / run_id
    if not path.is_dir():
        raise FileNotFoundError(f"Missing selected run: {path}. Restore artifacts or select MODE='train'.")
    return path


def verify_manifest(run):
    run = Path(run).resolve()
    checked, absent = [], []
    manifest = read_json(run / "manifest.json")
    if not manifest:
        raise ValueError("Empty manifest")
    for name, digest in manifest.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run):
            raise ValueError(f"Unsafe manifest path: {name}")
        if path.is_file():
            if hash_file(path) != digest:
                raise ValueError(f"SHA256 mismatch: {name}")
            checked.append(name)
        elif Path(name).parts[0] in {"best_model", "checkpoints"}:
            absent.append(name)
        else:
            raise FileNotFoundError(f"Required evidence missing: {name}")
    if read_json(run / "status.json")["status"] != "completed":
        raise ValueError("Selected run did not complete")
    return {"verified_files": len(checked), "omitted_checkpoint_files": absent,
            "checkpoint_available_locally": (run / "best_model/model.safetensors").is_file()}


def audit_exercise1(run):
    run = Path(run)
    result = verify_manifest(run)
    breakdown = read_json(run / "tokenizer_breakdown.json")
    ids, tokens, mask = (breakdown[k] for k in ("input_ids", "tokens", "attention_mask"))
    if not len(ids) == len(tokens) == len(mask) or ids[0] != 101 or ids[-1] != 102:
        raise ValueError("Invalid tokenizer evidence")
    if not set(mask).issubset({0, 1}):
        raise ValueError("Invalid attention mask")
    rows = read_json(run / "predictions.json")
    if len(rows["pipeline"]) != len(rows["manual"]) or not rows["manual"]:
        raise ValueError("Missing inference comparisons")
    for high, low in zip(rows["pipeline"], rows["manual"]):
        if high["label"] != low["label"] or not np.isclose(high["score"], low["score"], atol=1e-5):
            raise ValueError("Pipeline/manual mismatch")
        if not np.isclose(sum(low["probabilities"]), 1, atol=1e-6):
            raise ValueError("Invalid inference probabilities")
    return dict(result, run=run.name, compared_sentences=len(rows["manual"]))


def audit_finetuning(run):
    run = Path(run)
    result = verify_manifest(run)
    stats = read_json(run / "data_statistics.json")
    if stats["split_sizes"] != {"train": 8530, "validation": 1066, "test": 1066}:
        raise ValueError("Dataset split contract differs")
    rows = read_json(run / "all_predictions.json")
    if len(rows) != 1066 or [r["index"] for r in rows] != list(range(1066)):
        raise ValueError("Missing, reordered or duplicate test indices")
    probs = np.array([r["probabilities"] for r in rows], dtype=float)
    labels = np.array([r["true_label"] for r in rows])
    preds = np.array([r["predicted_label"] for r in rows])
    if (probs.shape != (1066, 2) or not np.isfinite(probs).all()
        or np.any((probs < 0) | (probs > 1))
        or not np.allclose(probs.sum(1), 1, atol=1e-6)):
        raise ValueError("Invalid probability vectors")
    if not np.array_equal(np.bincount(labels, minlength=2), [533, 533]):
        raise ValueError("Unexpected test label counts")
    if not np.array_equal(preds, probs.argmax(1)):
        raise ValueError("Stored predictions differ from probabilities")
    metrics = compute_metrics((probs, labels))
    saved = read_json(run / "test_metrics.json")
    if saved.keys() != metrics.keys() or any(not np.isclose(saved[k], v, atol=1e-12, rtol=0) for k, v in metrics.items()):
        raise ValueError("Stored metrics differ from full predictions")
    for sample in read_json(run / "sample_predictions.json"):
        original = rows[sample["index"]]
        if any(sample[k] != original[k] for k in ["text", "true_label", "predicted_label"]):
            raise ValueError("Sample prediction differs from full predictions")
        if not np.isclose(sample["confidence"], max(original["probabilities"]), atol=1e-7):
            raise ValueError("Sample confidence differs")
    history = read_json(run / "trainer_history.json")
    validation = [row for row in history if "eval_accuracy" in row]
    config = read_json(run / "config.json")
    train = read_json(run / "train_metrics.json")
    if len(validation) != config["exercise2"]["num_train_epochs"] or train["epoch"] != len(validation):
        raise ValueError("Incomplete training history")
    reload = read_json(run / "reload_check.json")
    if reload.get("weights_exact_match") is not True:
        raise ValueError("Missing saved checkpoint verification evidence")
    baseline = compute_metrics((np.tile([1.0, 0.0], (len(labels), 1)), labels))
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return dict(result, run=run.name, metrics=metrics, constant_negative_baseline=baseline, correct=int(np.trace(cm)),
                incorrect=int(cm.sum() - np.trace(cm)), confusion_matrix=cm.tolist(),
                best_validation_epoch=max(validation, key=lambda row: row["eval_accuracy"])["epoch"],
                classification_report=classification_report(labels, preds, target_names=["NEGATIVE", "POSITIVE"], output_dict=True),
                checkpoint_verification="Kaggle reload_check.json; weights are not re-executed by this audit")
