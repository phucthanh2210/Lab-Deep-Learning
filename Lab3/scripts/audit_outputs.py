"""Audit selected experiment outputs without GPU, network or model weights."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence import selected_run, audit_exercise1, audit_finetuning
from src.utils.runs import save_json, ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation/artifact_audit.json")
    args = parser.parse_args()
    result = {"exercise1": audit_exercise1(selected_run("exercise1")),
              "finetuning": audit_finetuning(selected_run("finetuning"))}
    save_json(args.output, result)
    print("Artifact audit: PASSED")
    print("Verified files:", sum(row["verified_files"] for row in result.values()))
    print("Recomputed test accuracy:", result["finetuning"]["metrics"]["accuracy"])
    print("Checkpoint availability:", result["finetuning"]["checkpoint_available_locally"])
    print("Audit report:", args.output)
