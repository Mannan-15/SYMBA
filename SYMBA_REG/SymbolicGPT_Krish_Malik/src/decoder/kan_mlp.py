import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from efficient_kan import KAN # Make sure you pip installed efficient-kan

# 1. Generate Synthetic Physics Data: Damped Harmonic Oscillator
# Equation: y = e^(-0.1 * x) * sin(3 * x)
torch.manual_seed(42)
X = torch.linspace(0, 10, 500).unsqueeze(1)
y_true = torch.exp(-0.1 * X) * torch.sin(3 * X)
y_noisy = y_true + torch.randn_like(y_true) * 0.05 # Add 5% sensor noise

# 2. Define standard PyTorch MLP (Baseline)
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(), # Tanh is usually best for smooth physics curves
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 3. Initialize Models & Optimizers
mlp_model = StandardMLP()
kan_model = KAN([1, 8, 1]) # KAN needs fewer hidden parameters

criterion = nn.MSELoss()
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.01)
optimizer_kan = optim.Adam(kan_model.parameters(), lr=0.01)

# 4. Fast Training Loop
epochs = 300
mlp_losses, kan_losses = [], []

for epoch in range(epochs):
    # MLP Step
    optimizer_mlp.zero_grad()
    loss_mlp = criterion(mlp_model(X), y_noisy)
    loss_mlp.backward()
    optimizer_mlp.step()
    mlp_losses.append(loss_mlp.item())
    
    # KAN Step
    optimizer_kan.zero_grad()
    loss_kan = criterion(kan_model(X), y_noisy)
    loss_kan.backward()
    optimizer_kan.step()
    kan_losses.append(loss_kan.item())

# 5. Plot the Loss Curves to prove KAN superiority
plt.figure(figsize=(10, 5))
plt.plot(mlp_losses, label="Standard MLP (Baseline)", color='red', linestyle='dashed')
plt.plot(kan_losses, label="Kolmogorov-Arnold Network (Proposed)", color='green', linewidth=2)
plt.yscale('log') # Log scale shows the convergence difference perfectly
plt.title("Training Loss: MLP vs KAN on Damped Harmonic Oscillator")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()