#!/usr/bin/env python3
"""
🔧 Force GPU Usage - Kaggle Fix Script
Run this BEFORE your training code
"""

import os
import torch
import gc

print("🔧 FORCING GPU USAGE")
print("=" * 40)

# 1. Set environment variables
print("1️⃣ Setting CUDA environment...")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both T4 GPUs  
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 2. Clear all caches
print("2️⃣ Clearing caches...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

# Clear Python garbage
gc.collect()

# 3. Force CUDA initialization
print("3️⃣ Initializing CUDA...")
if torch.cuda.is_available():
    # Initialize all available GPUs
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        # Create a small tensor to initialize the context
        _ = torch.zeros(1, device=f'cuda:{i}')
        print(f"   ✅ GPU {i} initialized: {torch.cuda.get_device_name(i)}")
    
    # Set primary device
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print(f"   🎯 Primary device: {device}")
else:
    print("   ❌ CUDA not available!")
    device = torch.device('cpu')

# 4. Verify GPU is working
print("4️⃣ Testing GPU...")
if torch.cuda.is_available():
    test_tensor = torch.randn(1000, 1000, device=device)
    result = torch.mm(test_tensor, test_tensor)
    print(f"   ✅ GPU test passed: {result.device}")
    print(f"   📊 Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    print("   ❌ GPU test failed!")

# 5. Configure for optimal performance
print("5️⃣ Optimizing settings...")

# Set multiprocessing method
torch.multiprocessing.set_start_method('spawn', force=True)

# Enable optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.enabled = True

# Set tensor core precision
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

print("   ✅ Performance optimizations enabled")

# 6. Print final status
print("\n🏁 GPU SETUP COMPLETE")
print(f"Device: {device}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"CUDA Version: {torch.version.cuda}")

# Return the device for use in training
device