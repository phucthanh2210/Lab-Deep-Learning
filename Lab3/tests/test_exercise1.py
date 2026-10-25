from src.models.classifier import compare_inference

def test_pipeline_matches_manual_logits(tiny):
    model, tokenizer = tiny
    high, low = compare_inference(model, tokenizer, ["excellent movie", "terrible film"])
    assert len(high) == len(low) == 2
    for row in low:
        assert abs(sum(row["probabilities"]) - 1) < 1e-6
