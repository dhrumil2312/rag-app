import os
import json
from datasets import load_dataset

DATASET_NAME = "wikipedia"
SUBSET_NAME = "20220301.en"


SAMPLE_SIZE = 1_000_000
FILE_SIZE = 10_000


def download_wikipedia_dataset():
    print("Downloading dataset from hugging face")

    dataset = load_dataset(
        DATASET_NAME,
        SUBSET_NAME,
        split="train",
        cache_dir="./data",
        # streaming=True,
        # cache_dir="./data",
    )
    for i, row in enumerate(dataset):
        if i >= SAMPLE_SIZE:
            break
        data = {
            "id": row["id"],
            "url": row["url"],
            "title": row["title"],
            "text": row["text"],
        }

        if i % FILE_SIZE == 0:
            file_name = f"./json_files/wikipedia_{i // FILE_SIZE}.jsonl"
            print(f"Creating file: {file_name}")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "a") as f:
            f.write(json.dumps(data) + "\n")

def main():
    dataset = download_wikipedia_dataset()

if __name__ == "__main__":
    main()

