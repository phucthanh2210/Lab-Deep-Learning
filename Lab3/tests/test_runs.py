import hashlib
import json
import pytest
from src.utils.runs import run_context

def test_runs_preserve_history_and_verify_manifest(tmp_path):
    with run_context("test", {"seed": 42}, root=tmp_path) as first:
        print("logged evidence")
        (first / "metrics.json").write_text('{"accuracy": 1}')
    with run_context("test", {"seed": 42}, root=tmp_path) as second:
        print("new run")
    assert first != second and first.exists()
    assert "logged evidence" in (first / "console.log").read_text()
    for name, digest in json.loads((first / "manifest.json").read_text()).items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    with pytest.raises(RuntimeError):
        with run_context("test", {}, root=tmp_path):
            raise RuntimeError("intentional test failure")
    state = json.loads((tmp_path / "config/workflow.json").read_text())
    assert state["active_runs"]["test"] == second.name
    assert [r["status"] for r in state["runs"]] == ["completed", "completed", "failed"]
