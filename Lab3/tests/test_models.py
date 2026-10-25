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
