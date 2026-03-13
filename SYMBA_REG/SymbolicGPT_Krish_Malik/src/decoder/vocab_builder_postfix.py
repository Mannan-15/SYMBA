import json
from pathlib import Path

# 1. Load your upgraded Postfix JSON
input_file = Path("./feynman_parse_trees_postfix.json")
with input_file.open("r") as f:
    data = json.load(f)

# 2. Extract the clean token arrays directly (No parsing needed!)
# Filter out any None values (failed parses)
tokenized_trees = [entry["symbolic_parse_tree"] for entry in data if entry.get("symbolic_parse_tree")]

# 3. Build the Vocabulary
vocab = {}
# Add special tokens first so they get IDs 0, 1, 2, 3
special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
for tok in special_tokens:
    vocab[tok] = len(vocab)

# Add all the mathematical tokens
for tokens in tokenized_trees:
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)

print(f"Vocabulary size: {len(vocab)} tokens")

# 4. Convert token arrays to ID sequences (with <SOS> and <EOS>)
tokenized_id_seqs = []
for tokens in tokenized_trees:
    # Get ID, fallback to <UNK> if somehow missing
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens] 
    
    # Wrap sequence in Start and End tags
    final_sequence = [vocab["<SOS>"]] + ids + [vocab["<EOS>"]]
    tokenized_id_seqs.append(final_sequence)

# 5. Save the final Model-Ready JSON
output = {
    "vocab": vocab,
    "tokenized_trees": tokenized_id_seqs
}

output_file = Path("./tokenized_gpt_labels_postfix.json")
with output_file.open("w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Model-ready token IDs saved to {output_file}")
