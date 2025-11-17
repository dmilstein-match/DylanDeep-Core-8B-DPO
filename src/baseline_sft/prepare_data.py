import os
import json
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)

    # GSM8K main config: train + test
    ds = load_dataset("gsm8k", "main")
    train = ds["train"]
    test = ds["test"]

    # Simple 80/20 split of train into train/dev
    split_idx = int(0.8 * len(train))
    train_split = train.select(range(split_idx))
    dev_split = train.select(range(split_idx, len(train)))

    def dump_jsonl(path, dataset):
        with open(path, "w", encoding="utf-8") as f:
            for x in dataset:
                f.write(json.dumps(x) + "\n")

    dump_jsonl("data/gsm8k_train.jsonl", train_split)
    dump_jsonl("data/gsm8k_dev.jsonl", dev_split)
    dump_jsonl("data/gsm8k_test.jsonl", test)

    print("Saved GSM8K train/dev/test JSONL files in data/")

if __name__ == "__main__":
    main()
