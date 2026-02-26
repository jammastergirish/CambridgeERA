# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
# ]
# ///

from datasets import concatenate_datasets, load_dataset
import os


NUM_SAMPLES = 10000
BIO_RETAIN_RATIO = 0.25
SEED = 42


def create_forget_set(outpath: str, num_samples: int = NUM_SAMPLES) -> int:
    """Download WMDP-Bio forget corpus and write forget-set text file.

    Source: cais/wmdp-bio-forget-corpus (gated, requires HF login).

    Returns the number of samples written.
    """
    print("[create_datasets] Downloading WMDP-Bio forget corpus...")
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")

    texts = []
    for ex in ds:
        t = ex["text"].strip()
        if t:
            texts.append(t)

    texts = texts[:num_samples]

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")

    print(f"[create_datasets] ✓ Wrote {len(texts)} forget samples to {outpath}")
    return len(texts)


def create_retain_set(outpath: str, num_samples: int = NUM_SAMPLES) -> int:
    """Download WikiText-103 + WMDP bio-retain-corpus and write retain-set text file.

    Follows Cas's training code: select num_samples from wikitext-103,
    add num_samples * 0.25 from bio-retain-corpus, concatenate, shuffle,
    and take num_samples total (~80% wikitext, ~20% bio-retain).

    Sources:
      - wikitext / wikitext-103-raw-v1 (train split)
      - cais/wmdp-corpora / bio-retain-corpus (train split)

    Returns the number of samples written.
    """
    # 1. WikiText-103: filter empty, then select num_samples
    print("[create_datasets] Downloading WikiText-103 for retain set...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    wiki = wiki.filter(lambda ex: len(ex["text"].strip()) > 0)
    wiki = wiki.shuffle(seed=SEED).select(range(min(num_samples, len(wiki))))
    print(f"[create_datasets]   WikiText-103: {len(wiki)} samples")

    # 2. WMDP bio-retain-corpus: select num_samples * BIO_RETAIN_RATIO
    print("[create_datasets] Downloading WMDP bio-retain-corpus for retain set...")
    bio_retain = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
    n_bio = int(num_samples * BIO_RETAIN_RATIO)
    bio_retain = bio_retain.shuffle(seed=SEED).select(range(min(n_bio, len(bio_retain))))
    print(f"[create_datasets]   Bio-retain: {len(bio_retain)} samples")

    # 3. Concatenate, shuffle, select num_samples
    combined = concatenate_datasets([wiki, bio_retain]).shuffle(seed=SEED)
    combined = combined.select(range(min(num_samples, len(combined))))
    print(f"[create_datasets]   Combined retain: {len(combined)} samples")

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        for ex in combined:
            t = ex["text"].replace("\n", " ").strip()
            if t:
                f.write(t + "\n")

    n_written = sum(1 for _ in open(outpath))
    print(f"[create_datasets] ✓ Wrote {n_written} retain samples to {outpath}")
    return n_written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create forget/retain datasets.")
    parser.add_argument("--only", choices=["forget", "retain"], help="Create only the specified dataset.")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.only != "retain":
        create_forget_set(os.path.join(args.outdir, "forget.txt"), args.num_samples)
    if args.only != "forget":
        create_retain_set(os.path.join(args.outdir, "retain.txt"), args.num_samples)

    print("[create_datasets] Done.")
