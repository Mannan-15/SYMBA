import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Importing the JEPA modules
from models.jepa_modules import JEPA_PhysicsEncoder, JEPA_MathEncoder, LatentPredictor

# 1. Dataset Loader
class JEPADataset(Dataset):
    def __init__(self, tnet_json_path, postfix_json_path):
        print(f"Loading T-Net embeddings from: {tnet_json_path}")
        print(f"Loading Postfix tokens from: {postfix_json_path}")
        
        try:
            with open(tnet_json_path, 'r') as f:
                self.tnet_data = json.load(f)
            with open(postfix_json_path, 'r') as f:
                self.postfix_data = json.load(f)
                
            # Convert to tensors
            self.x_tensors = torch.tensor(self.tnet_data, dtype=torch.float32)
            self.y_tensors = torch.tensor(self.postfix_data, dtype=torch.long)
            self.valid = True
            print(f"Successfully loaded {len(self.x_tensors)} pairs.")
            
        except Exception as e:
            print(f"JSON Structure mismatch or file not found. Error: {e}")
            print("Falling back to simulated data for PoC demonstration...")
            self.x_tensors = torch.randn(100, 128) # Simulated 128D T-Net embeddings
            self.y_tensors = torch.randint(1, 50, (100, 15)) # Simulated Postfix sequences
            self.valid = False

    def __len__(self):
        return len(self.x_tensors)

    def __getitem__(self, idx):
        return self.x_tensors[idx], self.y_tensors[idx]

# 2. The VICReg Anti-Collapse Loss Function
def vicreg_loss(s_x, s_y, var_weight=1.0, inv_weight=1.0, gamma=1.0):
    """
    Forces s_x and s_y to match, while preventing the KAN layers from 
    collapsing to zero by enforcing a minimum variance (gamma).
    """
    # A. Invariance Loss (Make the embeddings match)
    inv_loss = F.mse_loss(s_x, s_y)
    
    # B. Variance Loss (Prevent collapse)
    std_x = torch.sqrt(s_x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(s_y.var(dim=0) + 1e-04)
    var_loss = torch.mean(F.relu(gamma - std_x)) + torch.mean(F.relu(gamma - std_y))
    
    return (inv_weight * inv_loss) + (var_weight * var_loss), inv_loss.item(), var_loss.item()

# 3. The Main Training Loop
def run_jepa_poc():
    TNET_PATH = "./src/embeddings/tnet_embeddings_my.json"
    POSTFIX_PATH = "./src/labels/tokenized_gpt_labels_postfix.json"
    
    dataset = JEPADataset(TNET_PATH, POSTFIX_PATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # JEPA + KAN models
    physics_encoder = JEPA_PhysicsEncoder(tnet_dim=128, latent_dim=64)
    math_encoder = JEPA_MathEncoder(vocab_size=100, latent_dim=64)
    predictor = LatentPredictor(latent_dim=64)
    
    # Add predictor parameters to the optimizer
    optimizer = torch.optim.AdamW(
        list(physics_encoder.parameters()) + 
        list(math_encoder.parameters()) + 
        list(predictor.parameters()), 
        lr=1e-3
    )
    
    print("\n--- Starting LM-JEPA Cross-Modal Alignment ---")
    n_epochs = 100
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        total_inv = 0
        total_var = 0
        cos_sim = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # 1. Encoders generate latent representations
            s_x = physics_encoder(batch_x) # Context
            s_y = math_encoder(batch_y) # Target
            s_pred = predictor(s_x) # Predictor guesses the target from the context
            
            # 2. VICReg Loss
            loss, inv, var = vicreg_loss(s_pred, s_y)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            total_inv += inv
            total_var += var
            # Cosine Similarity
            cos_sim += F.cosine_similarity(s_pred, s_y).mean().item()
            
        if (epoch+1) % 10 == 0:
            avg_loss = total_epoch_loss / len(dataloader)
            avg_inv = total_inv / len(dataloader)
            avg_var = total_var / len(dataloader)
            avg_cos = cos_sim / len(dataloader)
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | MSE: {avg_inv:.4f} | Var: {avg_var:.4f} | Cosine Sim: {avg_cos:.4f}")

if __name__ == "__main__":
    run_jepa_poc()