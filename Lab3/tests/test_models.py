import numpy as np
from src.models.classifier import load_model, manual_predict

def test_checkpoint_reload_preserves_predictions(tiny, tmp_path):
    model, tok = tiny
    before = manual_predict(model, tok, ["excellent movie"])
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    loaded, loaded_tok = load_model(str(tmp_path / "model"))
    after = manual_predict(loaded, loaded_tok, ["excellent movie"])
    assert np.allclose(before[0]["probabilities"], after[0]["probabilities"])
    assert loaded.config.num_labels == 2


def test_reload_check_ignores_forward_wrappers_but_checks_weights(tiny, tmp_path):
    import torch
    import pytest
    from src.models.classifier import verify_saved_model
    model, tok = tiny
    target = tmp_path / "saved"
    model.save_pretrained(target); tok.save_pretrained(target)
    original_forward = model.forward
    def wrapped_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        result.logits = result.logits + torch.tensor([0.1, -0.1])
        return result
    model.forward = wrapped_forward
    result = verify_saved_model(model, target, ["excellent movie"])
    assert result["weights_exact_match"]
    with torch.no_grad():
        model.classifier.bias[0] += 1
    with pytest.raises(AssertionError, match="weight differs"):
        verify_saved_model(model, target, ["excellent movie"])
