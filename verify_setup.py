#!/usr/bin/env python3
"""
Verification script to check project structure is set up correctly.
This runs in Replit to verify everything before pushing to GitHub.
"""
import os
import sys

def check_file(path, description):
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ MISSING {description}: {path}")
        return False

def main():
    print("=" * 60)
    print("Looper-Math-Platinum-8B - Project Structure Verification")
    print("=" * 60)
    
    all_good = True
    
    print("\n📁 Checking directories...")
    dirs = [
        ("data/", "Data directory"),
        ("checkpoints/", "Checkpoints directory"),
        ("src/", "Source directory"),
        ("src/baseline_sft/", "Baseline SFT module"),
        ("src/rl_training/", "RL training module"),
        ("src/regime_w/", "Regime W module (PRIVATE)"),
        ("src/eval/", "Evaluation module"),
    ]
    
    for path, desc in dirs:
        all_good &= check_file(path, desc)
    
    print("\n📄 Checking core files...")
    files = [
        ("requirements.txt", "Dependencies file"),
        ("README.md", "Documentation"),
        ("src/__init__.py", "Source package marker"),
    ]
    
    for path, desc in files:
        all_good &= check_file(path, desc)
    
    print("\n🔬 Checking baseline SFT module...")
    sft_files = [
        ("src/baseline_sft/__init__.py", "SFT package marker"),
        ("src/baseline_sft/prepare_data.py", "Data preparation script"),
        ("src/baseline_sft/train_sft.py", "SFT training script"),
    ]
    
    for path, desc in sft_files:
        all_good &= check_file(path, desc)
    
    print("\n🤖 Checking RL training module...")
    rl_files = [
        ("src/rl_training/__init__.py", "RL package marker"),
        ("src/rl_training/collect_rollouts.py", "Rollout collection (placeholder)"),
        ("src/rl_training/train_ppo.py", "PPO training (placeholder)"),
    ]
    
    for path, desc in rl_files:
        all_good &= check_file(path, desc)
    
    print("\n🔒 Checking Regime W module (PRIVATE)...")
    regime_files = [
        ("src/regime_w/__init__.py", "Regime W package marker"),
        ("src/regime_w/arms.py", "Multi-armed bandit (placeholder)"),
        ("src/regime_w/scoring.py", "Coherence scoring (placeholder)"),
        ("src/regime_w/reward.py", "Reward computation (placeholder)"),
        ("src/regime_w/demo.py", "Demo script (placeholder)"),
    ]
    
    for path, desc in regime_files:
        all_good &= check_file(path, desc)
    
    print("\n📊 Checking evaluation module...")
    eval_files = [
        ("src/eval/__init__.py", "Eval package marker"),
        ("src/eval/eval_platinum.py", "GSM8K-Platinum eval (placeholder)"),
    ]
    
    for path, desc in eval_files:
        all_good &= check_file(path, desc)
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All files and directories are in place!")
        print("\n📌 Next steps:")
        print("1. Commit and push to GitHub:")
        print("   git add .")
        print("   git commit -m 'Initial project scaffold'")
        print("   git push")
        print("\n2. On Lambda GPU:")
        print("   git clone <your-repo-url>")
        print("   cd DylanDeep-Core-8B-DPO")
        print("   pip install -r requirements.txt")
        print("   python -m src.baseline_sft.prepare_data")
        print("   python -m src.baseline_sft.train_sft")
        print("=" * 60)
        return 0
    else:
        print("❌ Some files or directories are missing!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
