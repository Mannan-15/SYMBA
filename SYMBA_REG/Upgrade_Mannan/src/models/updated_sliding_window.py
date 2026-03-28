import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA (Local Postfix Files)
# ==========================================
with open("./src/embeddings/tnet_embeddings_my.json", "r") as f:
    embeddings_data = json.load(f)
with open("./src/labels/tokenized_gpt_labels_postfix.json", "r") as f:
    label_data = json.load(f)

X = [torch.tensor(e["embedding"], dtype=torch.float32) for e in embeddings_data]
Y = label_data["tokenized_trees"]
vocab = label_data["vocab"]
inv_vocab = {v: k for k, v in vocab.items()}

pad_token_id = vocab["<PAD>"]
eos_token_id = vocab["<EOS>"]
# sos_token_id = vocab["<SOS>"]

# ==========================================
# 2. DATASET 
# ==========================================
class SymbolicDataset(Dataset):
    def __init__(self, embeddings, token_seqs, pad_token_id, max_len=None):
        self.embeddings = embeddings
        self.token_seqs = token_seqs
        self.pad_token_id = pad_token_id
        self.max_len = max_len or max(len(seq) for seq in token_seqs)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        y = self.token_seqs[idx]
        # Pad the sequence
        y_padded = y + [self.pad_token_id] * (self.max_len - len(y))
        return {
            "embedding": x,
            "target_ids": torch.tensor(y_padded, dtype=torch.long),
        }

dataset = SymbolicDataset(X, Y, pad_token_id)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ==========================================
# 3. SPARSE ATTENTION (Standard)
# ==========================================
class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, num_random=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.window_size = window_size
        self.num_random = num_random

    def forward(self, x):
        B, T, D = x.shape
        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.window_size)
            mask[i, start:i+1] = 0
            if self.num_random > 0:
                rand_idx = torch.randint(0, T, (self.num_random,), device=x.device)
                mask[i, rand_idx] = 0
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_out

# ==========================================
# 4. SYMBOLIC DECODER (Absolute Positions)
# ==========================================
class SymbolicDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, gpt_dim=256, n_layers=3, n_heads=4, max_len=100):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, gpt_dim)
        
        # Restored standard positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, gpt_dim))
        
        self.embedding_proj = nn.Linear(embedding_dim, gpt_dim)
        self.dropout = nn.Dropout(0.4)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "sparse_attn": SparseAttention(gpt_dim, n_heads),
                "ff": nn.Sequential(
                    nn.Linear(gpt_dim, 4 * gpt_dim),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(4 * gpt_dim, gpt_dim)
                ),
                "norm1": nn.LayerNorm(gpt_dim),
                "norm2": nn.LayerNorm(gpt_dim),
            }) for _ in range(n_layers)
        ])
        self.out = nn.Linear(gpt_dim, vocab_size)

    def forward(self, embedding, target_ids):
        B, T = target_ids.shape
        
        # Shift input for Teacher Forcing
        input_ids = target_ids[:, :-1] if T > 1 else target_ids
        
        tok_embed = self.dropout(self.token_embedding(input_ids))
        context_tok = self.embedding_proj(embedding).unsqueeze(1)
        x = torch.cat([context_tok, tok_embed], dim=1)
        
        # Add absolute positional embeddings
        x = x + self.pos_embedding[:, :x.shape[1], :]
        
        for layer in self.layers:
            attn_out = layer["sparse_attn"](x)
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)
            
        logits = self.out(x[:, 1:, :]) 
        return logits

# ==========================================
# 5. TRAINING LOOP WITH VISUALIZATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SymbolicDecoder(
    vocab_size=len(vocab),
    embedding_dim=128,
    gpt_dim=256,
    n_layers=3,
    n_heads=4,
    max_len=dataset.max_len
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

num_epochs = 50

print(f"Starting Training for {num_epochs} epochs on {device}...")

for epoch in range(num_epochs):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for batch in train_loader:
        embedding = batch["embedding"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        logits = model(embedding, target_ids)
        # Shifted targets: Model predicts everything after <SOS>
        shifted_targets = target_ids[:, 1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = logits.argmax(dim=-1)
        mask = (shifted_targets != pad_token_id)
        total_correct += ((preds == shifted_targets) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    avg_train_acc = (total_correct / total_tokens)*100
    train_accs.append(avg_train_acc)

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_tokens = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            embedding = batch["embedding"].to(device)
            target_ids = batch["target_ids"].to(device)
            logits = model(embedding, target_ids)
            shifted_targets = target_ids[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1))
            val_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            mask = (shifted_targets != pad_token_id)
            val_correct += ((preds == shifted_targets) & mask).sum().item()
            val_tokens += mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = (val_correct / val_tokens) * 100
    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)
    
    ppl_train = math.exp(avg_train_loss)
    ppl_val = math.exp(avg_val_loss)

    scheduler.step(avg_val_loss)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} (PPL: {ppl_train:.2f}) | Val Loss: {avg_val_loss:.4f} (PPL: {ppl_val:.2f}) || Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}%")
    
# ==========================================
# 6. Plot Loss and Accuracy on Dual Axes
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss on left Y-axis
color1 = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss', color=color1)
ax1.plot(train_losses, label="Train Loss", color='red', linestyle='dashed')
ax1.plot(val_losses, label="Validation Loss", color='darkred', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)
ax1.set_title("Model Loss")
ax1.legend(loc="upper right")

# --- Plot 2: Accuracy (Right Subplot) ---
color2 = 'tab:blue'
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Token Accuracy (%)', color=color2)  
ax2.plot(train_accs, label="Train Accuracy", color='dodgerblue', linestyle='dashed')
ax2.plot(val_accs, label="Validation Accuracy", color='blue', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(True, alpha=0.3)
ax2.set_title("Next-Token Accuracy")
ax2.legend(loc="lower right")

fig.tight_layout()  
fig.suptitle("Training Loss vs. Next-Token Accuracy")
plt.show()

torch.save(model.state_dict(), "./src/checkpoints/symbolic_gpt_decoder_sparse_regularized.pth")

# ==========================================
# 6. BEAM SEARCH INFERENCE
# ==========================================

def generate_next_token(model, embedding, generated_ids):
    """
    generated_ids: [1, t] — tokens generated so far (empty at first step)
    Returns logits for the next token: [vocab_size]
    """
    if generated_ids.shape[1] == 0:
        # First step: only the context token exists, no input tokens
        context_tok = model.embedding_proj(embedding).unsqueeze(1)
        x = context_tok + model.pos_embedding[:, :1, :]
        for layer in model.layers:
            attn_out = layer["sparse_attn"](x)
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)
        return model.out(x[:, 0, :]).squeeze(0)  # [vocab_size]
    else:
        # Subsequent steps: context + generated tokens so far
        tok_embed = model.token_embedding(generated_ids)
        context_tok = model.embedding_proj(embedding).unsqueeze(1)
        x = torch.cat([context_tok, tok_embed], dim=1)
        x = x + model.pos_embedding[:, :x.shape[1], :]
        for layer in model.layers:
            attn_out = layer["sparse_attn"](x)
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)
        logits = model.out(x[:, 1:, :])
        return logits[:, -1, :].squeeze(0)  # last position [vocab_size]


def beam_search_decode(model, embedding, eos_id, beam_width=5, max_len=60):
    model.eval()
    # Start with empty generated sequence — no PAD needed
    beams = [(torch.zeros(1, 0, dtype=torch.long, device=device), 0.0)]

    with torch.no_grad():
        for step in range(max_len):
            all_candidates = []
            for seq, score in beams:
                # Stop expanding finished beams
                if seq.shape[1] > 0 and seq[0, -1].item() == eos_id:
                    all_candidates.append((seq, score))
                    continue

                log_probs = F.log_softmax(
                    generate_next_token(model, embedding, seq), dim=-1
                )
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_tok = topk_ids[i].view(1, 1)
                    new_seq = torch.cat([seq, next_tok], dim=1)
                    new_score = score + topk_log_probs[i].item()
                    all_candidates.append((new_seq, new_score))

            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop if all beams have finished
            if all(b[0].shape[1] > 0 and b[0][0, -1].item() == eos_id for b in beams):
                break

    return beams[0][0].squeeze(0)  # best beam's token ids


# ── Inference loop ──────────────────────────────────────────────────────────
def decode_tokens(token_ids):
    tokens = []
    for t in token_ids:
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        if t_val == eos_token_id:
            break
        if t_val != pad_token_id:
            tokens.append(inv_vocab[t_val])
    return " ".join(tokens)

predictions = []
exact_matches = 0
total_samples = 0

model.eval()
for idx, batch in enumerate(val_loader):
    embedding = batch["embedding"].to(device)
    target_ids = batch["target_ids"].to(device)

    pred_ids = beam_search_decode(model, embedding, eos_id=eos_token_id, beam_width=5)

    pred_expr = decode_tokens(pred_ids)
    true_expr = decode_tokens(target_ids.squeeze(0))

    is_match = pred_expr.replace(" ", "") == true_expr.replace(" ", "")
    if is_match:
        exact_matches += 1
    total_samples += 1

    predictions.append({
        "id": idx,
        "prediction": pred_expr,
        "ground_truth": true_expr,
        "exact_match": is_match
    })

accuracy = (exact_matches / total_samples) * 100
print(f"EXACT MATCH: {accuracy:.2f}% ({exact_matches}/{total_samples})")

failures = [p for p in predictions if not p["exact_match"]]
print(f"\n❌ Sample Failures ({len(failures)} total):")
for p in failures[:5]:
    print(f"  PRED : {p['prediction']}")
    print(f"  TRUTH: {p['ground_truth']}")
    print()

with open("./data/predictions_beam.json", "w") as f:
    json.dump(predictions, f, indent=2)
print("Predictions saved to predictions_beam.json")
