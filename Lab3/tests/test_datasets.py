from datasets import Dataset, DatasetDict
from src.datasets.loader import load_data, tokenize_data, EXPECTED_SIZES

def test_official_split_contract(monkeypatch):
    ds = DatasetDict({k: Dataset.from_dict({"text": ["movie"] * n,
                     "label": [i % 2 for i in range(n)]}) for k, n in EXPECTED_SIZES.items()})
    monkeypatch.setattr("src.datasets.loader.load_dataset", lambda *a, **kw: ds)
    result = load_data({"dataset_name": "fixture", "dataset_revision": "fixture"})
    assert {k: len(result[k]) for k in result} == EXPECTED_SIZES
    import pytest
    ds["train"] = ds["train"].select(range(10))
    with pytest.raises(ValueError, match="split sizes"):
        load_data({"dataset_name": "fixture", "dataset_revision": "fixture"})

def test_tokenization_and_dynamic_padding(tiny):
    from transformers import DataCollatorWithPadding
    _, tokenizer = tiny
    ds = DatasetDict(train=Dataset.from_dict({"text": ["movie", "i loved this excellent movie"], "label": [0, 1]}))
    result = tokenize_data(ds, tokenizer, max_length=5)["train"]
    assert "text" not in result.column_names
    assert len(result[0]["input_ids"]) == 3
    assert len(result[1]["input_ids"]) == 5
    batch = DataCollatorWithPadding(tokenizer)([result[0], result[1]])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["labels"].tolist() == [0, 1]
