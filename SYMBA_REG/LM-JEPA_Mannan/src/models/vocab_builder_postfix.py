import json
import re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
CONST_TOKEN = "<C>"   # replaces ALL numeric literals

def is_numeric(tok):
    """Return True if token is a raw number (int or float, pos or neg)."""
    try:
        float(tok)
        return True
    except ValueError:
        return False

# ── 1. Load postfix parse trees ──────────────────────────────────────────────
input_file = Path("./src/parser/feynman_parse_trees_postfix.json")
with input_file.open("r") as f:
    data = json.load(f)

valid_entries = [e for e in data if e.get("symbolic_parse_tree")]
raw_trees     = [e["symbolic_parse_tree"] for e in valid_entries]

# ── 2. Normalize: replace every numeric token with <C> ───────────────────────
def normalize(tokens):
    return [CONST_TOKEN if is_numeric(tok) else tok for tok in tokens]

normalized_trees = [normalize(tree) for tree in raw_trees]

# ── 3. Build vocabulary ───────────────────────────────────────────────────────
# NOTE: No <SOS> — the embedding vector serves as the start signal
special_tokens = ["<PAD>", "<EOS>", "<UNK>", CONST_TOKEN]
vocab = {tok: i for i, tok in enumerate(special_tokens)}

for tokens in normalized_trees:
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)

print(f"Vocabulary size: {len(vocab)} tokens")
print(f"Sample tokens: {list(vocab.keys())[:20]}")

# ── 4. Tokenize to ID sequences ───────────────────────────────────────────────
eos_id = vocab["<EOS>"]
unk_id = vocab["<UNK>"]

tokenized_id_seqs = []
for tokens in normalized_trees:
    ids = [vocab.get(tok, unk_id) for tok in tokens]
    ids.append(eos_id)          # append EOS at end (no SOS at start)
    tokenized_id_seqs.append(ids)

# ── 5. Save ───────────────────────────────────────────────────────────────────
output = {
    "vocab": vocab,
    "tokenized_trees": tokenized_id_seqs,

    # Bonus: save original float values alongside, indexed by sample
    # Useful if you later want to reconstruct exact expressions
    "const_values": [
        [tok for tok in tree if is_numeric(tok)]
        for tree in raw_trees
    ]
}

output_file = Path("./src/labels/tokenized_gpt_labels_postfix.json")
with output_file.open("w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Saved to {output_file}")

## What the output now looks like
# Before:  [1, 4, 5, 6, ...]     where 4 = "-1.0", 5 = "3.14159", ...
# After:   [q, m, <C>, mul, eos] where <C> covers ALL constants
