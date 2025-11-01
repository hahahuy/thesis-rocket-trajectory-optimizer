#!/bin/bash

# PINN Training Script for Rocket Trajectory Optimization
# This script demonstrates a simple Physics-Informed Neural Network training

set -e  # Exit on any error

echo "🚀 Starting PINN Training Demo for Rocket Trajectory Optimization"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data/mini
mkdir -p logs
mkdir -p checkpoints

echo "📊 Creating mini dataset..."

# Create a simple trajectory dataset using Python
python3 << 'EOF'
import numpy as np
import h5py
import os

# Create a small trajectory dataset
print("Generating mini trajectory dataset...")

# Time vector
t = np.linspace(0, 10, 100)

# Simple projectile motion with drag
g = 9.81
v0 = 100.0
angle = 45.0
drag_coeff = 0.01

v0x = v0 * np.cos(np.radians(angle))
v0y = v0 * np.sin(np.radians(angle))

# Simple integration (Euler method)
dt = t[1] - t[0]
x = np.zeros_like(t)
y = np.zeros_like(t)
vx = np.zeros_like(t)
vy = np.zeros_like(t)

x[0] = 0
y[0] = 0
vx[0] = v0x
vy[0] = v0y

for i in range(1, len(t)):
    # Simple dynamics with drag
    v = np.sqrt(vx[i-1]**2 + vy[i-1]**2)
    ax = -drag_coeff * v * vx[i-1]
    ay = -g - drag_coeff * v * vy[i-1]
    
    vx[i] = vx[i-1] + ax * dt
    vy[i] = vy[i-1] + ay * dt
    x[i] = x[i-1] + vx[i-1] * dt
    y[i] = y[i-1] + vy[i-1] * dt

# Save dataset
os.makedirs('data/mini', exist_ok=True)
with h5py.File('data/mini/trajectory.h5', 'w') as f:
    f.create_dataset('time', data=t)
    f.create_dataset('position', data=np.column_stack([x, y]))
    f.create_dataset('velocity', data=np.column_stack([vx, vy]))
    f.create_dataset('acceleration', data=np.column_stack([ax, ay]))

print(f"✅ Dataset created: {len(t)} time points")
print(f"   Position range: x=[{x.min():.1f}, {x.max():.1f}], y=[{y.min():.1f}, {y.max():.1f}]")
print(f"   Velocity range: vx=[{vx.min():.1f}, {vx.max():.1f}], vy=[{vy.min():.1f}, {vy.max():.1f}]")
EOF

echo "🧠 Starting PINN training..."

# Simple PINN training using Python
python3 << 'EOF'
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

print("Loading dataset...")

# Load the dataset
with h5py.File('data/mini/trajectory.h5', 'r') as f:
    t = torch.tensor(f['time'][:], dtype=torch.float32)
    pos = torch.tensor(f['position'][:], dtype=torch.float32)
    vel = torch.tensor(f['velocity'][:], dtype=torch.float32)

# Normalize data
t_norm = (t - t.min()) / (t.max() - t.min())
pos_norm = (pos - pos.mean(dim=0)) / pos.std(dim=0)
vel_norm = (vel - vel.mean(dim=0)) / vel.std(dim=0)

print(f"Dataset loaded: {len(t)} points")

# Simple PINN model
class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 4)  # [x, y, vx, vy]
        )
    
    def forward(self, t):
        return self.net(t.unsqueeze(-1))

# Create model and optimizer
model = SimplePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Starting training...")

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    pred = model(t_norm)
    pred_pos = pred[:, :2]
    pred_vel = pred[:, 2:]
    
    # Data loss
    data_loss = criterion(pred_pos, pos_norm) + criterion(pred_vel, vel_norm)
    
    # Physics loss (simplified)
    # For this demo, we'll use a simple constraint
    physics_loss = torch.mean((pred_vel[:, 1] + 9.81 * t)**2)  # Gravity constraint
    
    total_loss = data_loss + 0.1 * physics_loss
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.6f} "
              f"(Data: {data_loss.item():.6f}, Physics: {physics_loss.item():.6f})")

print("✅ Training completed!")

# Save model
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/pinn_model.pth')
print("💾 Model saved to checkpoints/pinn_model.pth")

# Create a simple plot
print("📊 Creating visualization...")
with torch.no_grad():
    pred = model(t_norm)
    pred_pos = pred[:, :2] * pos.std(dim=0) + pos.mean(dim=0)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(pos[:, 0].numpy(), pos[:, 1].numpy(), 'b-', label='True trajectory', linewidth=2)
plt.plot(pred_pos[:, 0].numpy(), pred_pos[:, 1].numpy(), 'r--', label='PINN prediction', linewidth=2)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t.numpy(), pos[:, 1].numpy(), 'b-', label='True Y', linewidth=2)
plt.plot(t.numpy(), pred_pos[:, 1].numpy(), 'r--', label='PINN Y', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
plt.title('Y Position vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/pinn_training_demo.png', dpi=150, bbox_inches='tight')
plt.close()

print("📈 Visualization saved to figures/pinn_training_demo.png")
print("🎉 PINN training demo completed successfully!")
EOF

echo ""
echo "✅ PINN Training Demo Completed Successfully!"
echo "============================================="
echo "📁 Generated files:"
echo "   - data/mini/trajectory.h5 (mini dataset)"
echo "   - checkpoints/pinn_model.pth (trained model)"
echo "   - figures/pinn_training_demo.png (visualization)"
echo ""
echo "🚀 The demo shows:"
echo "   1. Dataset generation with physics-based trajectory"
echo "   2. PINN model training with data and physics losses"
echo "   3. Model evaluation and visualization"
echo ""
echo "This demonstrates the core workflow for your thesis project!"
