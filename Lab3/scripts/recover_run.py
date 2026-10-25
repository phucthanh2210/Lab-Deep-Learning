"""Recover a completed training run that failed only at the old reload assertion."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.workflow import recover_reload_failure
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(recover_reload_failure(args.run_dir))
