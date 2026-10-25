from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.workflow import finetune
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    print(finetune(allow_cpu=args.allow_cpu))
