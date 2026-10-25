import numpy as np
from src.utils.plots import plot_lengths, plot_learning, plot_confusion, plot_metrics, plot_samples

def test_five_plots_written(tmp_path):
    plot_lengths([3, 4, 5, 6, 7], 5, tmp_path / "token_length_distribution.png")
    plot_learning([{"epoch": 1, "loss": .6}, {"epoch": 1, "eval_loss": .5, "eval_accuracy": .8}], tmp_path / "learning_curves.png")
    plot_confusion([0, 0, 1, 1], [0, 1, 1, 1], tmp_path / "confusion_matrix.png")
    plot_metrics({"accuracy": .75, "binary_precision": .66, "binary_recall": 1, "binary_f1": .8}, tmp_path / "metrics_bar.png")
    plot_samples([{"text": "Example review", "true_label": 0, "predicted_label": 1, "confidence": .9}], tmp_path / "sample_predictions.png")
    images = list(tmp_path.glob("*.png"))
    assert len(images) == 5
    assert all(p.stat().st_size > 1000 for p in images)
