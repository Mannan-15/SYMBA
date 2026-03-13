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
with open("./tnet_embeddings.json", "r") as f:
    embeddings_data = json.load(f)
with open("./tokenized_gpt_labels_postfix.json", "r") as f:
    label_data = json.load(f)

X = [torch.tensor(e["embedding"], dtype=torch.float32) for e in embeddings_data]
Y = label_data["tokenized_trees"]
vocab = label_data["vocab"]
inv_vocab = {v: k for k, v in vocab.items()}

pad_token_id = vocab["<PAD>"]
eos_token_id = vocab["<EOS>"]
sos_token_id = vocab["<SOS>"]

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
num_epochs = 80

print(f"🚀 Starting Training for {num_epochs} epochs on {device}...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
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

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            embedding = batch["embedding"].to(device)
            target_ids = batch["target_ids"].to(device)
            logits = model(embedding, target_ids)
            shifted_targets = target_ids[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), shifted_targets.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

# --- PLOT LOSS CURVES ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", color='blue', lw=2)
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color='orange', linestyle='--', lw=2)
plt.yscale('log') # Log scale helps see the drop from 2.0 to 0.0003 much better
plt.xlabel("Epochs")
plt.ylabel("Loss (Log Scale)")
plt.title("Convergence Profile: Postfix Symbolic Transformer")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig("training_convergence.png") # Saves a high-res version for your proposal
plt.show()

torch.save(model.state_dict(), "symbolic_gpt_baseline_beam.pth")
print("✅ Model weights and Loss Plot saved.")

# ==========================================
# 6. BEAM SEARCH INFERENCE
# ==========================================
def decode_tokens(token_ids):
    tokens = []
    for t in token_ids:
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        if t_val != pad_token_id and t_val != sos_token_id:
            tokens.append(inv_vocab[t_val])
    if "<EOS>" in tokens:
        tokens = tokens[:tokens.index("<EOS>")]
    return " ".join(tokens)

def beam_search_decode(model, embedding, sos_id, eos_id, beam_width=5, max_len=60):
    model.eval()
    start_seq = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    beams = [(start_seq, 0.0)]
    
    with torch.no_grad():
        for step in range(max_len):
            all_candidates = []
            for seq, score in beams:
                if seq[0, -1].item() == eos_id:
                    all_candidates.append((seq, score))
                    continue
                
                logits = model(embedding, seq) 
                next_token_logits = logits[:, -1, :] 
                log_probs = F.log_softmax(next_token_logits, dim=-1).squeeze(0)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = topk_ids[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[i].item()
                    all_candidates.append((new_seq, new_score))
            
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            beams = ordered[:beam_width]
            if beams[0][0][0, -1].item() == eos_id:
                break
                
    return beams[0][0].squeeze(0)

predictions = []
for idx, batch in enumerate(val_loader):
    embedding = batch["embedding"].to(device)
    target_ids = batch["target_ids"].to(device)
    
    pred_ids = beam_search_decode(model, embedding, sos_token_id, eos_token_id, beam_width=5)
    
    pred_expr = decode_tokens(pred_ids)
    true_expr = decode_tokens(target_ids.squeeze(0))
    
    predictions.append({
        "id": idx,
        "prediction": pred_expr,
        "ground_truth": true_expr
    })

with open("./predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("✅ Advanced Beam Search Predictions saved to predictions.json")