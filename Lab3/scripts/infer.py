from pathlib import Path
import sys
import argparse
import json
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.workflow import read_config
from src.models.classifier import load_model, manual_predict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    cfg = read_config()
    if args.model_path and not args.model_path.is_dir():
        parser.error("Checkpoint directory does not exist; refusing to fall back to an untuned model")
    ex1 = cfg["exercise1"]
    model, tok = load_model(str(args.model_path) if args.model_path else ex1["model_name"],
                           None if args.model_path else ex1["revision"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(manual_predict(model, tok, [args.text]), indent=2))
