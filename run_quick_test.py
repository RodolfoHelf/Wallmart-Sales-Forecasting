#!/usr/bin/env python3
"""
Quick Test Launcher Script
Runs the organized quick test from the project root
"""

import os
import sys
import subprocess

def main():
    """Launch the quick test from the organized scripts folder"""
    print("🏪 Walmart Sales Forecasting - Quick Test Launcher")
    print("=" * 50)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the organized quick test script
    quick_test_script = os.path.join(project_root, "models", "quick_test.py")
    
    if not os.path.exists(quick_test_script):
        print(f"❌ Error: Quick test script not found at {quick_test_script}")
        return 1
    
    print(f"📁 Project Root: {project_root}")
    print(f"⚡ Launching Quick Test: {quick_test_script}")
    print("=" * 50)
    
    try:
        # Change to project root and run the quick test
        os.chdir(project_root)
        
        # Run the quick test script
        result = subprocess.run([sys.executable, quick_test_script], 
                              cwd=project_root, 
                              check=True)
        
        print("\n✅ Quick test completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Quick test failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error launching quick test: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
