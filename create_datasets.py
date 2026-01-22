# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
# ]
# ///

from datasets import load_dataset

ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")

# Just take the questions/prompts
texts = []
for ex in ds:
    # Depending on version, key may be "question" or "prompt"
    if "question" in ex:
        texts.append(ex["question"])
    elif "prompt" in ex:
        texts.append(ex["prompt"])

texts = texts[:500]

with open("data/forget.txt", "w") as f:
    for t in texts:
        f.write(t.replace("\n", " ").strip() + "\n")


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

texts = []
for ex in ds:
    t = ex["text"].strip()
    if len(t) > 50:
        texts.append(t)

texts = texts[:500]

with open("data/retain.txt", "w") as f:
    for t in texts:
        f.write(t.replace("\n", " ").strip() + "\n")