# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
# ]
# ///

from datasets import load_dataset
import os


NUM_SAMPLES = 10000


def create_forget_set(outpath: str, num_samples: int = NUM_SAMPLES) -> int:
    """Download WMDP-Bio and write forget-set text file.

    Returns the number of samples written.
    """
    print("[create_datasets] Downloading WMDP-Bio dataset for forget set...")
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")

    texts = []
    for ex in ds:
        if "question" in ex:
            texts.append(ex["question"])
        elif "prompt" in ex:
            texts.append(ex["prompt"])

    texts = texts[:num_samples]

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")

    print(f"[create_datasets] ✓ Wrote {len(texts)} forget samples to {outpath}")
    return len(texts)


def create_retain_set(outpath: str, num_samples: int = NUM_SAMPLES, min_length: int = 50) -> int:
    """Download WikiText-2 and write retain-set text file.

    Returns the number of samples written.
    """
    print("[create_datasets] Downloading WikiText-2 dataset for retain set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = []
    for ex in ds:
        t = ex["text"].strip()
        if len(t) > min_length:
            texts.append(t)

    texts = texts[:num_samples]

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")

    print(f"[create_datasets] ✓ Wrote {len(texts)} retain samples to {outpath}")
    return len(texts)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_forget_set("data/forget.txt")
    create_retain_set("data/retain.txt")
    print("[create_datasets] Done — datasets ready in data/")