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
    
class JEPA_PhysicsEncoder(nn.Module):
    """
    Takes pre-computed T-Net embeddings (s_x) and maps them into 
    the final Joint Latent Space using KAN.
    """
    def __init__(self, tnet_dim=128, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.projector = KAN([tnet_dim, hidden_dim, latent_dim])

    def forward(self, tnet_emb):
        return self.projector(tnet_emb)

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
        
        # efficient_kan mapping from GRU hidden state (64) to latent space (64)
        self.projector = KAN([hidden_dim, latent_dim])

    def forward(self, postfix_tokens):
        # postfix_tokens shape: (batch_size, seq_len)
        embedded = self.embedding(postfix_tokens)
        
        # Get the final hidden state from the GRU
        _, hidden = self.gru(embedded) # hidden shape: (1, batch_size, hidden_dim)
        
        # Squeeze the layer dimension and pass through KAN
        latent_math = self.projector(hidden.squeeze(0)) 
        return latent_math
    