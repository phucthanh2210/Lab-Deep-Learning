"""Exclusive run folders, atomic registry, environment evidence and SHA256 manifest."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
import torch

ROOT = Path(__file__).resolve().parents[2]

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)

def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def snapshot():
    return {"python": sys.version, "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "versions": {n: importlib.metadata.version(n) for n in
             ["torch", "transformers", "datasets", "evaluate", "accelerate", "numpy", "scikit-learn"]}}

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text)
        self.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self): return False

@contextmanager
def run_context(kind, config, root=ROOT):
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + kind + "_" + uuid.uuid4().hex[:8]
    path = root / "outputs" / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    save_json(path / "config.json", config)
    save_json(path / "environment_snapshot.json", snapshot())
    status = "failed"
    try:
        with (path / "console.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print("Run:", run_id)
                try:
                    yield path
                    status = "completed"
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    raise
    finally:
        save_json(path / "status.json", {"status": status})
        manifest = {str(p.relative_to(path)): hash_file(p)
                    for p in sorted(path.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        save_json(path / "manifest.json", manifest)
        # Serialize registry updates so simultaneous completed runs cannot drop entries.
        from filelock import FileLock
        registry = root / "config" / "workflow.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / "outputs" / ".registry.lock")):
            state = json.loads(registry.read_text()) if registry.exists() else {"active_runs": {}, "runs": []}
            state.setdefault("runs", []).append({"id": run_id, "kind": kind, "status": status})
            if status == "completed": state.setdefault("active_runs", {})[kind] = run_id
            save_json(registry, state)
