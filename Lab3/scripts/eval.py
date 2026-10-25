from pathlib import Path
import sys
import argparse
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.workflow import read_config, evaluate_model, seed_all
from src.models.classifier import load_model
from src.datasets.loader import tokenize_data, load_data
from src.utils.runs import run_context
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_path.is_dir(): parser.error("Fine-tuned checkpoint directory not found")
    saved_config = args.model_path.resolve().parent / "config.json"
    cfg = read_config(saved_config if saved_config.is_file() else None)
    seed_all(cfg["seed"])
    cfg["evaluated_checkpoint"] = str(args.model_path.resolve())
    with run_context("evaluation", cfg) as out:
        raw = load_data(cfg["exercise2"])
        model, tok = load_model(str(args.model_path))
        tokenized = tokenize_data(raw, tok, cfg["exercise2"]["max_length"])
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(out / "temporary"),
            per_device_eval_batch_size=cfg["exercise2"]["eval_batch_size"], report_to="none"),
            processing_class=tok, data_collator=DataCollatorWithPadding(tok))
        evaluate_model(trainer, tokenized["test"], raw["test"], out)
