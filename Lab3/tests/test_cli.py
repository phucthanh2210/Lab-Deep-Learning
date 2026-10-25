"""Checkpoint evaluation must use the original experiment, not mutable defaults."""
import json
from pathlib import Path
import runpy
import sys
from datasets import Dataset, DatasetDict
from src.utils.runs import ROOT, run_context, save_json
from src.workflow import read_config


def test_eval_and_infer_use_checkpoint_config_without_backbone_download(tiny, tmp_path, monkeypatch, capsys):
    model, tokenizer = tiny
    run = tmp_path / "training_run"
    checkpoint = run / "best_model"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    cfg = read_config()
    cfg["exercise2"].update(dataset_name="original-dataset", model_name="must-not-download", max_length=4)
    save_json(run / "config.json", cfg)
    ds = Dataset.from_dict({"text": ["i loved this excellent movie", "a terrible boring film"], "label": [1, 0]})
    def load_original(config):
        assert config["dataset_name"] == "original-dataset"
        return DatasetDict(train=ds, validation=ds, test=ds)
    monkeypatch.setattr("src.datasets.loader.load_data", load_original)
    def capture_run(kind, config):
        assert config["exercise2"]["max_length"] == 4
        return run_context(kind, config, root=tmp_path / "evaluation")
    monkeypatch.setattr("src.utils.runs.run_context", capture_run)
    monkeypatch.setattr(sys, "argv", ["eval.py", "--model-path", str(checkpoint)])
    runpy.run_path(str(ROOT / "scripts/eval.py"), run_name="__main__")
    state = json.loads((tmp_path / "evaluation/config/workflow.json").read_text())
    result = tmp_path / "evaluation/outputs/runs" / state["active_runs"]["evaluation"]
    assert len(json.loads((result / "all_predictions.json").read_text())) == 2
    assert (result / "test_metrics.json").is_file()
    from src.models.classifier import manual_predict
    def predict_with_original_limit(model, tokenizer, texts, max_length):
        assert max_length == 4
        return manual_predict(model, tokenizer, texts, max_length)
    monkeypatch.setattr("src.models.classifier.manual_predict", predict_with_original_limit)
    monkeypatch.setattr(sys, "argv", ["infer.py", "--model-path", str(checkpoint), "--text", "i loved this excellent movie"])
    runpy.run_path(str(ROOT / "scripts/infer.py"), run_name="__main__")
    assert '"probabilities"' in capsys.readouterr().out
