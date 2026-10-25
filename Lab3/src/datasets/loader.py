"""Load the complete official splits; never subsample or rebuild validation."""
from datasets import load_dataset

EXPECTED_SIZES = {"train": 8530, "validation": 1066, "test": 1066}

def load_data(config):
    ds = load_dataset(config["dataset_name"], revision=config["dataset_revision"])
    sizes = {name: len(ds[name]) for name in EXPECTED_SIZES}
    if sizes != EXPECTED_SIZES:
        raise ValueError(f"Unexpected official split sizes: {sizes}")
    for name in EXPECTED_SIZES:
        if set(ds[name]["label"]) != {0, 1}:
            raise ValueError(f"Invalid binary labels in {name}")
    return ds

def tokenize_data(ds, tokenizer, max_length):
    return ds.map(lambda batch: tokenizer(batch["text"], truncation=True,
                  max_length=max_length), batched=True, remove_columns=["text"])
