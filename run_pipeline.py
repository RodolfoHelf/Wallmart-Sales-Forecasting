#!/usr/bin/env python3
"""
Pipeline Launcher Script
Runs the organized pipeline from the project root
"""

import os
import sys
import subprocess

def main():
    """Launch the pipeline from the organized scripts folder""" 
    print("🏪 Walmart Sales Forecasting - Pipeline Launcher")
    print("=" * 50)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the organized pipeline script
    pipeline_script = os.path.join(project_root, "scripts", "run_full_pipeline.py")
    
    if not os.path.exists(pipeline_script):
        print(f"❌ Error: Pipeline script not found at {pipeline_script}")
        return 1
    
    print(f"📁 Project Root: {project_root}")
    print(f"🚀 Launching Pipeline: {pipeline_script}")
    print("=" * 50)
    
    try:
        # Change to project root and run the pipeline
        os.chdir(project_root)
        
        # Run the pipeline script
        result = subprocess.run([sys.executable, pipeline_script], 
                              cwd=project_root, 
                              check=True)
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error launching pipeline: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
