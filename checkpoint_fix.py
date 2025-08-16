#!/usr/bin/env python3
"""
🔧 PyTorch 2.6 Checkpoint Loading Fix
Handles the weights_only and safe globals issue
"""

import torch
import numpy as np
from datetime import datetime

print("🔧 PyTorch 2.6 Checkpoint Fix")
print("=" * 40)

# Fix 1: Add safe globals for numpy (if you trust the checkpoint)
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.ndarray,
    np.dtype,
    np.core.multiarray._reconstruct,
    np.core.multiarray.scalar,
])

print("✅ Added numpy safe globals")

# Fix 2: Modified checkpoint loading function
def load_checkpoint_safe(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Safe checkpoint loading for PyTorch 2.6+
    """
    print(f"📁 Loading checkpoint: {checkpoint_path}")
    
    try:
        # Try with weights_only=True first (safer)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        print("✅ Loaded with weights_only=True (safest)")
    except Exception as e:
        print(f"⚠️  weights_only=True failed: {str(e)[:100]}...")
        
        try:
            # Fallback to weights_only=False (if you trust the source)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            print("✅ Loaded with weights_only=False (trusted source)")
        except Exception as e2:
            print(f"❌ Both loading methods failed!")
            print(f"Error: {e2}")
            return 0, [], [], [], []
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model state loaded")
        except Exception as e:
            print(f"⚠️  Model state loading failed: {e}")
    
    # Load optimizer state (optional)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded")
        except Exception as e:
            print(f"⚠️  Optimizer state loading failed: {e}")
    
    # Load scheduler state (optional)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Scheduler state loaded")
        except Exception as e:
            print(f"⚠️  Scheduler state loading failed: {e}")
    
    # Load scaler state (optional)
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✅ Scaler state loaded")
        except Exception as e:
            print(f"⚠️  Scaler state loading failed: {e}")
    
    # Get resume info
    resume_epoch = checkpoint.get('epoch', 0) + 1
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    val_maes = checkpoint.get('val_maes', [])
    val_rmses = checkpoint.get('val_rmses', [])
    
    print(f"📊 Checkpoint info:")
    print(f"  Resume from epoch: {resume_epoch}")
    print(f"  Training history: {len(train_losses)} epochs")
    
    return resume_epoch, train_losses, val_losses, val_maes, val_rmses

# Fix 3: Safe checkpoint saving function
def save_checkpoint_safe(epoch, model, optimizer, scheduler, scaler, 
                        train_losses, val_losses, val_maes, val_rmses, 
                        config, filepath, is_best=False):
    """
    Safe checkpoint saving for PyTorch 2.6+
    """
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'val_rmses': val_rmses,
        'config': config,
        'pytorch_version': torch.__version__,
        'timestamp': str(datetime.now())
    }
    
    # Add optional components
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save checkpoint
    try:
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved: {filepath}")
        
        # Copy to working directory for easy download
        import shutil
        import os
        
        filename = os.path.basename(filepath)
        working_path = f'/kaggle/working/{filename}'
        
        try:
            shutil.copy(filepath, working_path)
            print(f"✅ Copied to: {working_path}")
        except:
            pass
            
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            shutil.copy(filepath, best_path)
            print(f"🏆 Best model saved: {best_path}")
            
    except Exception as e:
        print(f"❌ Checkpoint save failed: {e}")

# Test the fix
if __name__ == "__main__":
    print("\n🧪 Testing checkpoint fix...")
    
    # Create a dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test save
    print("\n📝 Testing save...")
    save_checkpoint_safe(
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        train_losses=[1.0, 0.8, 0.6],
        val_losses=[1.2, 0.9, 0.7],
        val_maes=[10.0, 8.0, 6.0],
        val_rmses=[12.0, 9.0, 7.0],
        config={'test': True},
        filepath='/tmp/test_checkpoint.pth'
    )
    
    # Test load
    print("\n📖 Testing load...")
    resume_epoch, train_losses, val_losses, val_maes, val_rmses = load_checkpoint_safe(
        '/tmp/test_checkpoint.pth',
        model,
        optimizer
    )
    
    print(f"✅ Test complete! Resume epoch: {resume_epoch}")
    
    print("\n" + "=" * 40)
    print("🎯 To use in your training:")
    print("1. Add the safe globals at the top of your script")
    print("2. Replace torch.load() with load_checkpoint_safe()")
    print("3. Replace torch.save() with save_checkpoint_safe()")