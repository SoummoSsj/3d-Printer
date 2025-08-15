#!/usr/bin/env python3
"""
🚨 Emergency GPU Test - Minimal Model to Force GPU Usage
"""
import torch
import torch.nn as nn
import time

# Force GPU setup
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔍 Device: {device}")
print(f"🔍 CUDA available: {torch.cuda.is_available()}")

class SimpleSpeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        # Process frames
        x = x.view(-1, 3, 160, 160)
        features = self.conv(x)
        features = features.view(batch_size, seq_len, -1)
        # Use last frame for speed
        speed = self.fc(features[:, -1])
        return speed.squeeze(-1)

def test_gpu_usage():
    print("\n🚀 Testing GPU Usage")
    
    # Create model
    model = SimpleSpeedNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"✅ Model device: {next(model.parameters()).device}")
    
    # Create fake data
    batch_size = 4
    seq_len = 6
    frames = torch.randn(batch_size, seq_len, 3, 160, 160, device=device, dtype=torch.float32)
    targets = torch.randn(batch_size, device=device, dtype=torch.float32) * 50 + 80
    
    print(f"✅ Data device: {frames.device}")
    print(f"✅ Data shape: {frames.shape}")
    
    # Training step
    model.train()
    start_time = time.time()
    
    for i in range(10):
        optimizer.zero_grad()
        
        # Force GPU computation
        with torch.cuda.amp.autocast():
            predictions = model(frames)
            loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        if i == 0:
            print(f"✅ Predictions device: {predictions.device}")
            print(f"✅ Loss device: {loss.device}")
        
        if i % 3 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")
    
    end_time = time.time()
    print(f"\n⏱️ Time for 10 steps: {end_time - start_time:.2f}s")
    print(f"🔥 GPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"🔥 GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    return True

if __name__ == "__main__":
    try:
        success = test_gpu_usage()
        if success:
            print("\n✅ GPU Test PASSED - GPU is working!")
        else:
            print("\n❌ GPU Test FAILED")
    except Exception as e:
        print(f"\n💥 GPU Test ERROR: {e}")
        import traceback
        traceback.print_exc()