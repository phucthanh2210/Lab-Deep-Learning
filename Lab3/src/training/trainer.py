"""Trainer setup: CUDA-only FP16 and consistent epoch save/evaluation."""
import torch
from copy import copy
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from .metrics import compute_metrics

def make_trainer(model, tokenizer, ds, config, run_path):
    args = TrainingArguments(
        output_dir=str(run_path / "checkpoints"),
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"],
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        greater_is_better=True, save_total_limit=2,
        fp16=bool(config["fp16"] and torch.cuda.is_available()),
        seed=config["seed"], data_seed=config["seed"], report_to="none")
    trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                   train_dataset=ds["train"], eval_dataset=ds["validation"],
                   data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    class TrainingMetrics(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            # evaluate() resets TrainerControl flags; preserve scheduled validation/save.
            scheduled = copy(control)
            trainer.evaluate(ds["train"], metric_key_prefix="train")
            return scheduled
    trainer.add_callback(TrainingMetrics())
    return trainer

def sanity_overfit(model, tokenizer, steps=30, lr=1e-3):
    """Test-only gate; caller supplies a disposable model, never the training model."""
    texts = ["I loved this excellent movie", "A terrible boring film"] * 4
    device = next(model.parameters()).device
    batch = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    labels = torch.tensor([1, 0] * 4, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return float(model(**batch, labels=labels).loss)
