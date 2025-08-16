#!/usr/bin/env python3
"""
🔍 Comprehensive GPU Diagnostic for Kaggle
Identify and fix GPU usage issues
"""

import os
import sys
import time
import subprocess
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler

print("🔍 KAGGLE GPU DIAGNOSTIC")
print("=" * 60)

# =============================================================================
# 1. System & Environment Check
# =============================================================================

print("\n1️⃣ SYSTEM ENVIRONMENT:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")

# Check environment variables
cuda_vars = {
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
    'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER', 'Not set'),
    'CUDA_PATH': os.environ.get('CUDA_PATH', 'Not set')
}

print(f"\n🔧 CUDA Environment Variables:")
for var, value in cuda_vars.items():
    print(f"  {var}: {value}")

# =============================================================================
# 2. CUDA & GPU Detection
# =============================================================================

print(f"\n2️⃣ CUDA & GPU DETECTION:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (runtime): {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n📱 GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Memory info
        try:
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  Allocated: {memory_allocated:.2f} GB")
            print(f"  Reserved: {memory_reserved:.2f} GB")
        except:
            print(f"  Memory info: Unable to get")
else:
    print("❌ CUDA not available!")
    print("\n🔧 Possible issues:")
    print("  1. GPU not enabled in Kaggle notebook settings")
    print("  2. CUDA drivers not properly installed")
    print("  3. PyTorch not compiled with CUDA support")

# =============================================================================
# 3. GPU Stress Test
# =============================================================================

print(f"\n3️⃣ GPU STRESS TEST:")

if torch.cuda.is_available():
    try:
        # Force device selection
        device = torch.device('cuda:0')
        print(f"✅ Device selected: {device}")
        
        # Clear cache
        torch.cuda.empty_cache()
        print(f"✅ GPU cache cleared")
        
        # Test tensor operations
        print(f"\n🧪 Testing basic tensor operations...")
        
        # Create test tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        start_time = time.time()
        z = torch.mm(x, y)  # Matrix multiplication
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"✅ Matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
        print(f"✅ Result shape: {z.shape}")
        print(f"✅ Result device: {z.device}")
        
        # Test autocast
        print(f"\n🧪 Testing mixed precision...")
        with autocast('cuda'):
            z_fp16 = torch.mm(x, y)
        print(f"✅ Mixed precision working: {z_fp16.dtype}")
        
        # Test neural network
        print(f"\n🧪 Testing neural network...")
        
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 10)
                
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x
        
        model = TestNet().to(device)
        input_tensor = torch.randn(4, 3, 224, 224, device=device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        torch.cuda.synchronize()
        nn_time = time.time() - start_time
        
        print(f"✅ Neural network forward pass: {nn_time:.4f} seconds")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output device: {output.device}")
        
        # Memory usage after operations
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\n💾 Memory after operations:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Skipping GPU tests - CUDA not available")

# =============================================================================
# 4. DataLoader & Training Test
# =============================================================================

print(f"\n4️⃣ DATALOADER & TRAINING TEST:")

if torch.cuda.is_available():
    try:
        from torch.utils.data import Dataset, DataLoader
        
        class DummyDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Simulate realistic data size
                image = torch.randn(6, 3, 160, 160)  # [T, C, H, W]
                target = torch.randn(1)
                return image, target
        
        print(f"🧪 Testing DataLoader with GPU...")
        
        dataset = DummyDataset(32)  # Small dataset for test
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((4, 4))
                self.lstm = nn.LSTM(32*4*4, 64, batch_first=True)
                self.fc = nn.Linear(64, 1)
                
            def forward(self, x):
                B, T, C, H, W = x.shape
                features = []
                for t in range(T):
                    feat = self.conv(x[:, t])  # [B, 32, H/2, W/2]
                    feat = self.pool(feat)     # [B, 32, 4, 4]
                    feat = feat.flatten(1)     # [B, 32*4*4]
                    features.append(feat)
                
                sequence = torch.stack(features, dim=1)  # [B, T, 32*4*4]
                lstm_out, _ = self.lstm(sequence)
                output = self.fc(lstm_out[:, -1])  # Use last timestep
                return output
        
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        scaler = GradScaler('cuda')
        
        print(f"✅ Model created and moved to GPU")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Training loop test
        model.train()
        total_time = 0
        num_batches = 0
        
        print(f"\n🔥 Testing training loop...")
        
        for batch_idx, (data, target) in enumerate(dataloader):
            start_time = time.time()
            
            # Move data to GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Synchronize to get accurate timing
            torch.cuda.synchronize()
            batch_time = time.time() - start_time
            total_time += batch_time
            num_batches += 1
            
            print(f"  Batch {batch_idx+1}: {batch_time:.4f}s, Loss: {loss.item():.4f}")
            
            # Test only a few batches
            if batch_idx >= 3:
                break
        
        avg_time = total_time / num_batches
        print(f"\n✅ Training test completed!")
        print(f"Average batch time: {avg_time:.4f} seconds")
        
        if avg_time > 2.0:
            print(f"⚠️  WARNING: Batch time is slow ({avg_time:.4f}s)")
            print(f"   Expected: ~0.1-0.5s for this simple model")
            print(f"   Possible issues:")
            print(f"     • Data loading bottleneck")
            print(f"     • CPU fallback happening")
            print(f"     • Memory transfer inefficiency")
        else:
            print(f"✅ Batch time looks good! GPU is working properly.")
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 5. Performance Comparison
# =============================================================================

print(f"\n5️⃣ CPU vs GPU PERFORMANCE COMPARISON:")

if torch.cuda.is_available():
    try:
        # Test matrix multiplication on CPU vs GPU
        size = 2000
        
        # CPU test
        print(f"\n🖥️  CPU Test (matrix {size}x{size}):")
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        print(f"  Time: {cpu_time:.4f} seconds")
        
        # GPU test
        print(f"\n🚀 GPU Test (matrix {size}x{size}):")
        x_gpu = torch.randn(size, size, device=device)
        y_gpu = torch.randn(size, size, device=device)
        
        start_time = time.time()
        z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"  Time: {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"\n⚡ GPU Speedup: {speedup:.2f}x")
        
        if speedup < 2.0:
            print(f"⚠️  WARNING: Low GPU speedup!")
            print(f"   Expected: 5-20x for T4 GPU")
            print(f"   Actual: {speedup:.2f}x")
        else:
            print(f"✅ Good GPU acceleration!")
            
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

# =============================================================================
# 6. Recommendations
# =============================================================================

print(f"\n6️⃣ RECOMMENDATIONS:")

if not torch.cuda.is_available():
    print(f"🚨 CRITICAL: CUDA not available!")
    print(f"📋 Fix steps:")
    print(f"  1. Go to Kaggle notebook settings")
    print(f"  2. Enable 'GPU' accelerator")
    print(f"  3. Restart kernel")
    print(f"  4. Run this diagnostic again")
    
elif torch.cuda.device_count() == 0:
    print(f"🚨 CRITICAL: No GPUs detected!")
    print(f"📋 Fix steps:")
    print(f"  1. Check Kaggle notebook accelerator settings")
    print(f"  2. Ensure GPU quota is available")
    print(f"  3. Try creating a new notebook")
    
else:
    print(f"✅ GPU hardware detected correctly")
    
    # Check if we had slow performance
    try:
        if 'avg_time' in locals() and avg_time > 1.0:
            print(f"⚠️  Performance issue detected!")
            print(f"📋 Optimization steps:")
            print(f"  1. Reduce batch size (try batch_size=2)")
            print(f"  2. Reduce image size (try 128x128)")
            print(f"  3. Reduce sequence length (try 4 frames)")
            print(f"  4. Set num_workers=0 (disable multiprocessing)")
            print(f"  5. Use smaller model architecture")
        else:
            print(f"✅ Performance looks good!")
            print(f"📋 Your model should train efficiently")
    except:
        pass

# =============================================================================
# 7. Quick Fix Script Generator
# =============================================================================

print(f"\n7️⃣ QUICK FIX SCRIPT:")
print(f"Copy and run this if you're still having issues:")

fix_script = '''
# GPU Fix Script - Run this in a new cell
import torch
import os

# Force CUDA device selection
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()

# Test GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    test_tensor = torch.randn(100, 100, device=device)
    print(f"✅ GPU working: {test_tensor.device}")
    
    # Memory check
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    print("❌ GPU still not working - contact Kaggle support")
'''

print(fix_script)

print(f"\n" + "=" * 60)
print(f"🏁 DIAGNOSTIC COMPLETE")
print(f"=" * 60)