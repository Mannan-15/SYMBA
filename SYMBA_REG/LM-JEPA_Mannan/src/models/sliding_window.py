import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load Data
with open("./src/embeddings/tnet_embeddings_my.json", "r") as f:
    embeddings_data = json.load(f)
with open("./src/labels/tokenized_gpt_labels_postfix.json", "r") as f:
    label_data = json.load(f)

X = [torch.tensor(e["embedding"], dtype=torch.float32) for e in embeddings_data]
Y = label_data["tokenized_trees"]
vocab = label_data["vocab"]

# Add PAD and EOS tokens
if "<PAD>" not in vocab:
    vocab["<PAD>"] = max(vocab.values()) + 1
pad_token_id = vocab["<PAD>"]

if "<EOS>" not in vocab:
    vocab["<EOS>"] = max(vocab.values()) + 1
eos_token_id = vocab["<EOS>"]

# Append EOS to every sequence
Y = [seq + [eos_token_id] for seq in Y]

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
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Sparse Attention
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

# GPT Decoder with Dropout
class SymbolicDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, gpt_dim=256, n_layers=3, n_heads=4, max_len=100):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, gpt_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, gpt_dim))
        self.embedding_proj = nn.Linear(embedding_dim, gpt_dim)
        self.dropout = nn.Dropout(0.4)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "sparse_attn": SparseAttention(gpt_dim, n_heads),
                "ff": nn.Sequential(
                    nn.Linear(gpt_dim, 4 * gpt_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.15),
                    nn.Linear(4 * gpt_dim, gpt_dim)
                ),
                "norm1": nn.LayerNorm(gpt_dim),
                "norm2": nn.LayerNorm(gpt_dim),
            }) for _ in range(n_layers)
        ])
        self.out = nn.Linear(gpt_dim, vocab_size)
    
    def forward(self, embedding, target_ids):
        B, T = target_ids.shape
        
        # SHIFT: The input to the transformer is all tokens except the last one
        input_ids = target_ids[:, :-1] 
        
        tok_embed = self.dropout(self.token_embedding(input_ids))
        context_tok = self.embedding_proj(embedding).unsqueeze(1)
        x = torch.cat([context_tok, tok_embed], dim=1)
        x = x + self.pos_embedding[:, :T, :] # Adjust position embedding length
        
        for layer in self.layers:
            attn_out = layer["sparse_attn"](x)
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)
            
        # The logits predict the NEXT token, starting from index 1 (ignoring context_tok)
        logits = self.out(x[:, 1:, :]) 
        return logits

# ==========================================
# Training Loop with Accuracy Tracking
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SymbolicDecoder(
    vocab_size=len(vocab), embedding_dim=128, gpt_dim=256, n_layers=3, n_heads=4, max_len=dataset.max_len
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

num_epochs = 50
for epoch in range(num_epochs):
    # --- TRAINING ---
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    
    for batch in train_loader:
        embedding = batch["embedding"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        logits = model(embedding, target_ids)
        shifted_targets = target_ids[:, 1:].contiguous()
        
        loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # NEW: Calculate Token Accuracy (ignoring padding)
        preds = logits.argmax(dim=-1)
        mask = (shifted_targets != pad_token_id)
        total_correct += ((preds == shifted_targets) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = (total_correct / total_tokens) * 100
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    # --- VALIDATION ---
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
            
            # NEW: Calculate Validation Accuracy
            preds = logits.argmax(dim=-1)
            mask = (shifted_targets != pad_token_id)
            val_correct += ((preds == shifted_targets) & mask).sum().item()
            val_tokens += mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = (val_correct / val_tokens) * 100
    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)
    
    ppl_train = np.exp(avg_train_loss)
    ppl_val = np.exp(avg_val_loss)

    scheduler.step(avg_val_loss)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} (PPL: {ppl_train:.2f}) | Val Loss: {avg_val_loss:.4f} (PPL: {ppl_val:.2f}) || Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}%")

# ==========================================
# Plot Loss and Accuracy
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

# Plot Accuracy on right Y-axis
color2='tab:blue'
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Token Accuracy (%)', color=color2)  
ax2.plot(train_accs, label="Train Accuracy", color='dodgerblue', linestyle='dashed')
ax2.plot(val_accs, label="Validation Accuracy", color='blue', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(True, alpha=0.3)
ax2.set_title("Next-Token Accuracy")
ax2.legend(loc="lower right")

fig.tight_layout()  
fig.suptitle("Postfix Notation + <C> Tokenization", fontsize=14, fontweight='bold')
plt.show()

torch.save(model.state_dict(), "./src/checkpoints/symbolic_gpt_decoder_sparse_regularized_my.pth")

"""## PREDICTIONS.json"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -----------------------------
# 1. Load dataset + vocab
# -----------------------------
with open("./src/labels/tokenized_gpt_labels_postfix.json", "r") as f:
    label_data = json.load(f)

vocab = label_data["vocab"]
inv_vocab = {v: k for k, v in vocab.items()}
token_seqs = label_data["tokenized_trees"]

# EOS/PAD ids
pad_token_id = vocab["<PAD>"]
eos_token_id = vocab["<EOS>"]

# -----------------------------
# 2. Load embeddings
# -----------------------------
with open("./src/embeddings/tnet_embeddings_my.json", "r") as f:
    embeddings_data = json.load(f)

X = [torch.tensor(e["embedding"], dtype=torch.float32) for e in embeddings_data]

# -----------------------------
# 3. Dataset class
# -----------------------------
class SymbolicDataset(torch.utils.data.Dataset):
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
        y_padded = y + [self.pad_token_id] * (self.max_len - len(y))
        return {
            "embedding": x,
            "target_ids": torch.tensor(y_padded, dtype=torch.long),
        }

dataset = SymbolicDataset(X, token_seqs, pad_token_id)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------------
# 4. Load model definition
# -----------------------------
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

class SymbolicDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, gpt_dim=256, n_layers=3, n_heads=4, max_len=100):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, gpt_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, gpt_dim))
        self.embedding_proj = nn.Linear(embedding_dim, gpt_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "sparse_attn": SparseAttention(gpt_dim, n_heads),
                "ff": nn.Sequential(
                    nn.Linear(gpt_dim, 4 * gpt_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.15),
                    nn.Linear(4 * gpt_dim, gpt_dim)
                ),
                "norm1": nn.LayerNorm(gpt_dim),
                "norm2": nn.LayerNorm(gpt_dim),
            }) for _ in range(n_layers)
        ])
        self.out = nn.Linear(gpt_dim, vocab_size)

    def forward(self, embedding, target_ids):
        B, T = target_ids.shape
        
        input_ids = target_ids[:, :-1]
        tok_embed = self.token_embedding(input_ids)
        context_tok = self.embedding_proj(embedding).unsqueeze(1)
        x = torch.cat([context_tok, tok_embed], dim=1)
        x = x + self.pos_embedding[:, :T, :]
        
        for layer in self.layers:
            attn_out = layer["sparse_attn"](x)
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)
        
        logits = self.out(x[:, 1:, :])
        return logits

# -----------------------------
# 5. Load trained weights with correct max_len
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("./src/checkpoints/symbolic_gpt_decoder_sparse_regularized_my.pth", map_location=device)
pos_embedding_shape = checkpoint['pos_embedding'].shape
saved_max_len = pos_embedding_shape[1] - 1

print(f"Saved model was trained with max_len: {saved_max_len}")
print(f"Current dataset max_len: {dataset.max_len}")

model = SymbolicDecoder(
    vocab_size=len(vocab),
    embedding_dim=128,
    gpt_dim=256,
    n_layers=3,
    n_heads=4,
    max_len=saved_max_len
).to(device)

model.load_state_dict(checkpoint)
model.eval()

if dataset.max_len != saved_max_len:
    print(f"Updating dataset max_len from {dataset.max_len} to {saved_max_len}")
    dataset.max_len = saved_max_len
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------------
# 6. Inference & Accuracy Calculation
# -----------------------------
def decode_tokens(token_ids):
    tokens = []
    for t in token_ids:
        if isinstance(t, torch.Tensor):
            t = t.item()
        if t != pad_token_id:
            tokens.append(inv_vocab[t])
    if "<EOS>" in tokens:
        tokens = tokens[:tokens.index("<EOS>")]
    return " ".join(tokens)

predictions = []
exact_matches = 0
total_samples = 0

print("Starting inference and calculating Exact Match Accuracy...")

with torch.no_grad():
    for idx, batch in enumerate(loader):
        embedding = batch["embedding"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Get predictions
        logits = model(embedding, target_ids)
        pred_ids = logits.argmax(-1).squeeze(0)
        
        # Shift targets for comparison
        shifted_targets = target_ids[:, 1:].contiguous().squeeze(0)

        # Decode to string expressions
        pred_expr = decode_tokens(pred_ids)
        true_expr = decode_tokens(shifted_targets)

        # EXACT MATCH LOGIC: We strip whitespace to ensure pure structural comparison
        if pred_expr.replace(" ", "") == true_expr.replace(" ", ""):
            exact_matches += 1
            
        total_samples += 1

        predictions.append({
            "id": idx,
            "prediction": pred_expr,
            "ground_truth": true_expr
        })

# Calculate final percentages
accuracy = (exact_matches / total_samples) * 100

print("========================================")
print(f"Inference Complete!")
print(f"EXACT MATCH ACCURACY: {accuracy:.2f}% ({exact_matches}/{total_samples} perfect equations)")
print("========================================")

# -----------------------------
# 7. Save predictions.json
# -----------------------------
with open("./data/predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("Predictions saved to predictions.json")