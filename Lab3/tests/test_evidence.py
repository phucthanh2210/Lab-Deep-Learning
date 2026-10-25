import json
import shutil
import pytest
from src.evidence import selected_run, audit_exercise1, audit_finetuning, verify_manifest
from src.utils.runs import hash_file, ROOT

REFERENCE = ROOT / "outputs/runs/20261025T181550Z_finetuning_e02bced7"
EXERCISE1 = ROOT / "outputs/runs/20261025T181532Z_exercise1_71da6f40"


def test_real_kaggle_evidence_recomputes_metrics_without_models():
    first = audit_exercise1(EXERCISE1)
    result = audit_finetuning(REFERENCE)
    assert first["verified_files"] == 6 and first["compared_sentences"] == 3
    assert result["verified_files"] >= 18
    assert result["confusion_matrix"] == [[444, 89], [82, 451]]
    assert result["correct"] == 895 and result["incorrect"] == 171
    assert result["best_validation_epoch"] == 3
    assert all(name.startswith(("best_model/", "checkpoints/")) for name in result["omitted_checkpoint_files"])


def test_audit_rejects_corruption_and_inconsistent_metrics(tmp_path):
    run = tmp_path / "copied"
    shutil.copytree(REFERENCE, run)
    path = run / "test_metrics.json"
    metrics = json.loads(path.read_text())
    metrics["accuracy"] = 1
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_manifest(run)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["test_metrics.json"] = hash_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metrics differ"):
        audit_finetuning(run)
    (run / "all_predictions.json").unlink()
    with pytest.raises(FileNotFoundError, match="Required evidence missing"):
        verify_manifest(run)
