import torch
import torch.nn as nn
from efficient_kan import KAN 

class LatentPredictor(nn.Module):
    """
    The 'P' in JEPA. Takes the context embedding (physics) and 
    predicts the latent representation of the target (math).
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        # A lightweight predictive MLP in the latent space
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, context_latent):
        return self.net(context_latent)
    
# class JEPA_PhysicsEncoder(nn.Module):
#     """
#     Takes pre-computed T-Net embeddings (s_x) and maps them into 
#     the final Joint Latent Space using KAN.
#     """
#     def __init__(self, tnet_dim=128, hidden_dim=64, latent_dim=64):
#         super().__init__()
#         self.projector = KAN([tnet_dim, hidden_dim, latent_dim])

#     def forward(self, tnet_emb):
#         return self.projector(tnet_emb)

# ====================
# Sparse Attention
# ====================
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

# =====================
# Transformer Block
# =====================
class StandardTransformerBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.sparse_attn = SparseAttention(dim, num_heads)
        
        # Standard MLP from your SymbolicDecoder
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out = self.sparse_attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# ======================
# The Context Encoder
# ======================
class JEPA_PhysicsEncoder(nn.Module):
    """
    Takes pre-computed T-Net embeddings, projects them into a sequence,
    and passes them through your standard Sparse Transformer.
    """
    def __init__(self, tnet_dim=128, latent_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latent_tokens = 4 # Split into a sequence of 4 for Attention
        
        self.latent_projector = nn.Linear(tnet_dim, latent_dim * self.num_latent_tokens)
        
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(dim=latent_dim, num_heads=n_heads) for _ in range(n_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, tnet_emb):
        B = tnet_emb.shape[0]
        # Project and reshape into sequence: [Batch, Tokens, Dim]
        x = self.latent_projector(tnet_emb).view(B, self.num_latent_tokens, self.latent_dim)
        
        for block in self.blocks:
            x = block(x)
            
        # Pool the sequence back into one 64D vector
        x = x.transpose(1, 2) 
        s_x = self.pool(x).squeeze(-1) 
        
        return s_x
    
class JEPA_MathEncoder(nn.Module):
    """
    Takes tokenized Postfix equations (y) and maps them into 
    the exact same Joint Latent Space (s_y).
    """
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # RNN to handle the sequence length of the Postfix tokens
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # 3. Standard MLP Projector (Replaces the KAN for the PoC)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )

    def forward(self, postfix_tokens):
        # postfix_tokens shape: (batch_size, seq_len)
        embedded = self.embedding(postfix_tokens)
        
        # Get the final hidden state from the GRU
        _, hidden = self.gru(embedded) # hidden shape: (1, batch_size, hidden_dim)
        
        s_y = self.projector(hidden.squeeze(0)) 
        return s_y
