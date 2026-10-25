"""Five evidence plots, written from the actual run only."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_lengths(lengths, cutoff, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(lengths, bins=40, kde=len(set(lengths)) > 1, ax=ax)
    ax.axvline(cutoff, color="red", linestyle="--", label=f"max_length={cutoff}")
    ax.set(xlabel="Tokens before truncation", ylabel="Training examples")
    ax.legend()
    finish(fig, path)

def plot_learning(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [("loss", "Training loss"), ("eval_loss", "Validation loss")]:
        rows = [r for r in history if key in r and "epoch" in r]
        axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", label=label)
    for key, label in [("train_accuracy", "Train accuracy"), ("eval_accuracy", "Validation accuracy")]:
        rows = [r for r in history if key in r]
        axes[1].plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", label=label)
    axes[1].legend()
    axes[0].legend(); axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    finish(fig, path)

def plot_confusion(labels, preds, path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float),
                     where=cm.sum(axis=1, keepdims=True) != 0)
    annot = np.array([[f"{cm[i,j]}\n{norm[i,j]:.1%}" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(norm, annot=annot, fmt="", vmin=0, vmax=1, cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set(xlabel="Predicted", ylabel="True", title="Test confusion matrix (row normalized)")
    finish(fig, path)

def plot_metrics(metrics, path):
    keys = ["accuracy", "binary_precision", "binary_recall", "binary_f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Accuracy", "Precision", "Recall", "F1"], [metrics[k] for k in keys])
    ax.bar_label(bars, fmt="%.3f"); ax.set(ylim=(0, 1.1))
    finish(fig, path)

def plot_samples(rows, path):
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * .85)))
    ax.axis("off")
    body = [[textwrap.fill(r["text"], 85), str(r["true_label"]), str(r["predicted_label"]),
             f"{r['confidence']:.3f}"] for r in rows]
    table = ax.table(cellText=body, colLabels=["Review", "True", "Predicted", "Confidence"],
                     colWidths=[.73, .08, .09, .1], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(8)
    for i, row in enumerate(rows, 1):
        for j in range(4):
            table[i, j].set_facecolor("#d9ead3" if row["true_label"] == row["predicted_label"] else "#f4cccc")
            table[i, j].set_height(.9 / (len(rows) + 1))
    finish(fig, path)
