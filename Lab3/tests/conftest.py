import pytest
import torch
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "loved", "this", "excellent", "movie", "a", "terrible", "boring", "film"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tokenizer = DistilBertTokenizerFast(vocab_file=str(path), do_lower_case=True)
    # DistilBERT does not accept token_type_ids.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    cfg = DistilBertConfig(vocab_size=len(vocab), n_layers=1, dim=32, hidden_dim=64,
                          n_heads=2, dropout=0, attention_dropout=0, seq_classif_dropout=0,
                          num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
                          label2id={"NEGATIVE": 0, "POSITIVE": 1})
    return DistilBertForSequenceClassification(cfg), tokenizer
