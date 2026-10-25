"""Local metrics avoid downloading executable metric scripts."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    result = {"accuracy": float(accuracy_score(labels, predictions))}
    for average in ("binary", "macro"):
        p, r, f, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0)
        result.update({f"{average}_precision": float(p), f"{average}_recall": float(r),
                       f"{average}_f1": float(f)})
    return result
