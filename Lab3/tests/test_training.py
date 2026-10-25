import pytest
import torch
from datasets import Dataset, DatasetDict
from src.training.metrics import compute_metrics
from src.training.trainer import sanity_overfit, make_trainer
from src.workflow import read_config

def test_one_batch_sanity_overfit(tiny):
    model, tok = tiny
    loss = sanity_overfit(model, tok, steps=80, lr=.005)
    assert loss < .05, loss

@pytest.mark.parametrize("fp16_requested", [False, True])
def test_trainer_train_evaluate_and_best_checkpoint(tiny, tmp_path, fp16_requested):
    model, tok = tiny
    encoded = tok(["excellent movie", "terrible film"] * 4, truncation=True)
    data = Dataset.from_dict({**encoded, "label": [1, 0] * 4})
    ds = DatasetDict(train=data, validation=data, test=data)
    cfg = dict(read_config()["exercise2"], seed=42, num_train_epochs=2, batch_size=4, eval_batch_size=4, fp16=fp16_requested)
    trainer = make_trainer(model, tok, ds, cfg, tmp_path)
    # Effective batch size scales with GPU count (Kaggle T4 x2 included).
    expected_steps = len(trainer.get_train_dataloader()) * 2
    result = trainer.train()
    assert result.global_step == expected_steps
    assert trainer.state.best_model_checkpoint is not None
    metrics = trainer.evaluate()
    assert "eval_macro_f1" in metrics
    assert trainer.args.fp16 == (fp16_requested and torch.cuda.is_available())
    expected = compute_metrics(([[3, 0], [0, 3]], [0, 1]))
    assert expected["accuracy"] == expected["macro_f1"] == 1


def test_complete_workflow_with_local_fixture(tiny, tmp_path, monkeypatch):
    import json
    from src import workflow
    from src.utils.runs import run_context
    model, tok = tiny
    model_path = tmp_path / "initial_model"
    model.save_pretrained(model_path); tok.save_pretrained(model_path)
    raw = Dataset.from_dict({"text": ["excellent movie", "terrible film"] * 4, "label": [1, 0] * 4})
    raw_ds = DatasetDict(train=raw, validation=raw, test=raw)
    from src.datasets.loader import tokenize_data
    tokens = tokenize_data(raw_ds, tok, 16)
    monkeypatch.setattr(workflow, "prepare_data", lambda config: (raw_ds, tok, tokens))
    monkeypatch.setattr(workflow, "run_context", lambda kind, cfg: run_context(kind, cfg, root=tmp_path))
    cfg = read_config()
    cfg["exercise2"].update(model_name=str(model_path), model_revision=None,
        num_train_epochs=1, batch_size=4, eval_batch_size=4)
    out = workflow.finetune(cfg, allow_cpu=True)
    assert json.loads((out / "status.json").read_text())["status"] == "completed"
    assert len(list(out.glob("*.png"))) == 5
    assert len(json.loads((out / "all_predictions.json").read_text())) == 8
    assert (out / "best_model/model.safetensors").is_file()
    assert (out / "reload_check.json").is_file()
    assert (out / "report.md").is_file()
