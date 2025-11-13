#!/usr/bin/env python3
"""
DISCOVERSE Submodules Setup Script

This script intelligently initializes only the submodules needed for your installed features.
Run this after installing optional dependencies to download required submodules.

Usage:
    python scripts/setup_submodules.py                    # Auto-detect and setup required submodules
    python scripts/setup_submodules.py --module lidar     # Setup specific module submodules
    python scripts/setup_submodules.py --all              # Setup all submodules
    python scripts/setup_submodules.py --list             # List all available submodules
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

# Mapping of feature modules to required submodules
MODULE_SUBMODULES = {
    'gaussian-rendering': ['submodules/diff-gaussian-rasterization'],
    'randomain': ['submodules/ComfyUI'],
    'act': ['policies/act'],
    'lidar': ['submodules/MuJoCo-LiDAR'],
    'rdt': ['submodules/lerobot'],
    'diffusion-policy': ['submodules/lerobot'],
    'urdf2mjcf' : ['submodules/urdf2mjcf'],
    'xml-editor': ['submodules/XML-Editor'],
}

# All available submodules
ALL_SUBMODULES = [
    'submodules/diff-gaussian-rasterization',
    'submodules/ComfyUI', 
    'policies/act',
    'submodules/MuJoCo-LiDAR',
    'submodules/lerobot',
    'submodules/urdf2mjcf',
    'submodules/XML-Editor'
]

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def is_submodule_initialized(submodule_path):
    """Check if a submodule is already initialized."""
    full_path = Path(submodule_path)
    if not full_path.exists():
        return False
    
    # Check if directory has content (not just empty)
    try:
        return len(list(full_path.iterdir())) > 0
    except:
        return False

def initialize_submodule(submodule_path):
    """Initialize a specific submodule."""
    if is_submodule_initialized(submodule_path):
        print(f"✓ {submodule_path} already initialized")
        return True
    
    print(f"🔄 Initializing {submodule_path}...")
    success, stdout, stderr = run_command(f"git submodule update --init {submodule_path}")
    
    if success:
        print(f"✅ Successfully initialized {submodule_path}")
        return True
    else:
        print(f"❌ Failed to initialize {submodule_path}: {stderr}")
        return False

def detect_installed_modules():
    """Detect which optional modules are installed."""
    installed_modules = []
    
    try:
        import pkg_resources
        
        # Get the discoverse package info
        try:
            pkg = pkg_resources.get_distribution('discoverse')
            # This is a simplified detection - in reality you might want to check 
            # the actual installed dependencies
        except:
            pass
            
        # Alternative: try importing specific modules to detect installation
        test_imports = {
            'gaussian-rendering': ['torch', 'diff_gaussian_rasterization'],
            'lidar': ['taichi'],
            'xml-editor': ['PyQt5'],
            'act': ['einops', 'transformers'],
            'randomain': ['diffusers'],
        }
        
        for module, imports in test_imports.items():
            try:
                for imp in imports:
                    __import__(imp)
                installed_modules.append(module)
            except ImportError:
                continue
                
    except ImportError:
        print("⚠️  Cannot detect installed modules automatically.")
        print("   Please specify modules manually or install all submodules.")
    
    return installed_modules

def setup_submodules_for_modules(modules):
    """Setup submodules for specified modules."""
    submodules_to_init = set()
    
    for module in modules:
        if module in MODULE_SUBMODULES:
            submodules_to_init.update(MODULE_SUBMODULES[module])
        else:
            print(f"⚠️  Unknown module: {module}")
    
    if not submodules_to_init:
        print("ℹ️  No submodules needed for specified modules.")
        return
    
    print(f"📦 Setting up submodules for modules: {', '.join(modules)}")
    print(f"   Required submodules: {', '.join(submodules_to_init)}")
    print()
    
    success_count = 0
    for submodule in submodules_to_init:
        if initialize_submodule(submodule):
            success_count += 1
    
    print(f"\n🎉 Successfully set up {success_count}/{len(submodules_to_init)} submodules!")

def setup_all_submodules():
    """Setup all submodules."""
    print("📦 Setting up all submodules...")
    print()
    
    success_count = 0
    for submodule in ALL_SUBMODULES:
        if initialize_submodule(submodule):
            success_count += 1
    
    print(f"\n🎉 Successfully set up {success_count}/{len(ALL_SUBMODULES)} submodules!")

def list_submodules():
    """List all available submodules and their status."""
    print("📋 Available Submodules:")
    print("=" * 60)
    
    for module, submodules in MODULE_SUBMODULES.items():
        print(f"\n🔧 {module}:")
        for submodule in submodules:
            status = "✅ Initialized" if is_submodule_initialized(submodule) else "⚪ Not initialized"
            print(f"   {submodule} - {status}")
    
    print(f"\n📊 Status: {sum(1 for s in ALL_SUBMODULES if is_submodule_initialized(s))}/{len(ALL_SUBMODULES)} submodules initialized")

def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    parser = argparse.ArgumentParser(description="DISCOVERSE Submodules Setup")
    parser.add_argument('--module', action='append', help='Setup submodules for specific module(s)')
    parser.add_argument('--all', action='store_true', help='Setup all submodules')
    parser.add_argument('--list', action='store_true', help='List all submodules and their status')
    
    args = parser.parse_args()
    
    # Check if we're in a git repository
    if not Path('.git').exists():
        print("❌ Error: Not in a git repository root. Please run from DISCOVERSE root directory.")
        sys.exit(1)
    
    print("🚀 DISCOVERSE Submodules Setup")
    print("=" * 50)
    
    if args.list:
        list_submodules()
    elif args.all:
        setup_all_submodules()
    elif args.module:
        setup_submodules_for_modules(args.module)
    else:
        # Auto-detect mode
        print("🔍 Auto-detecting installed modules...")
        installed = detect_installed_modules()
        
        if installed:
            print(f"📦 Detected installed modules: {', '.join(installed)}")
            setup_submodules_for_modules(installed)
        else:
            print("ℹ️  No modules detected automatically.")
            print("   Use --list to see available options or --all to setup everything.")
            print("   Example: python scripts/setup_submodules.py --module lidar gaussian-rendering")

if __name__ == "__main__":
    main() 