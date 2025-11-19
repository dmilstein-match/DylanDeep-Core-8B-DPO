#!/usr/bin/env python3
"""
Runtime patch to FORCE disable gradient checkpointing in Abel model
This overrides the model's internal gradient checkpointing
"""

def patch_train_script():
    """Generate a patched version that forces gradient checkpointing off"""
    
    patch_code = '''
# === GRADIENT CHECKPOINTING FIX START ===
# Force disable gradient checkpointing at model level
import torch

# Monkey-patch checkpoint function to be a no-op
def no_checkpoint(func, *args, **kwargs):
    """Replace gradient checkpointing with normal forward pass"""
    return func(*args, **kwargs)

# Override PyTorch's checkpoint
torch.utils.checkpoint.checkpoint = no_checkpoint

# Also override the checkpoint_sequential if it exists
if hasattr(torch.utils.checkpoint, 'checkpoint_sequential'):
    torch.utils.checkpoint.checkpoint_sequential = lambda *args, **kwargs: args[0](*args[1:])

print("⚠️ GRADIENT CHECKPOINTING FORCEFULLY DISABLED AT RUNTIME")
# === GRADIENT CHECKPOINTING FIX END ===

'''
    
    # Read the original script
    import os
    script_path = "src/baseline_sft/train_sft_abel.py"
    
    if not os.path.exists(script_path):
        print(f"Cannot find {script_path}")
        return None
    
    with open(script_path, 'r') as f:
        original = f.read()
    
    # Find where to insert the patch (after imports)
    import_section_end = 0
    lines = original.split('\n')
    
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('import') and not line.strip().startswith('from'):
            if i > 10:  # Make sure we're past the initial imports
                import_section_end = i
                break
    
    # Insert the patch
    lines.insert(import_section_end, patch_code)
    patched = '\n'.join(lines)
    
    # Also ensure gradient_checkpointing is False in config
    patched = patched.replace('gradient_checkpointing=True', 'gradient_checkpointing=False')
    patched = patched.replace('model.gradient_checkpointing_enable()', '# model.gradient_checkpointing_enable()  # DISABLED')
    
    # Save patched version
    output_path = "train_sft_abel_patched.py"
    with open(output_path, 'w') as f:
        f.write(patched)
    
    print(f"✅ Created patched script: {output_path}")
    return output_path

if __name__ == "__main__":
    print("=" * 60)
    print("AGGRESSIVE Gradient Checkpointing Override")
    print("=" * 60)
    print("\nThis will create a patched script that FORCES gradient")
    print("checkpointing to be disabled at the PyTorch level.\n")
    
    patched_script = patch_train_script()
    
    if patched_script:
        print("\nRun the patched script with:")
        print(f"  torchrun --nproc_per_node=8 {patched_script}")
        print("\nOr test single GPU first:")
        print(f"  CUDA_VISIBLE_DEVICES=0 python {patched_script}")
