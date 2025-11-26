"""
Quick start script to run the complete pipeline
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 80)
    print(f"▶ {description}")
    print("=" * 80)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error running: {description}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def main():
    """Run the complete MLOps pipeline."""
    print("=" * 80)
    print("SERVICE TIME PREDICTION - QUICK START")
    print("=" * 80)
    
    # Check if data exists
    data_dir = Path("data/raw")
    required_files = [
        "orders.parquet",
        "articles.parquet",
        "service_times.parquet",
        "driver_order_mapping.parquet"
    ]
    
    print("\n📊 Checking data files...")
    missing_files = []
    for file in required_files:
        filepath = data_dir / file
        if filepath.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ Missing data files. Please add the following files to data/raw/:")
        for file in missing_files:
            print(f"   - {file}")
        return
    
    # Step 1: Train model
    if not run_command("python train.py", "Training Model"):
        return
    
    # Step 2: Start API (in background is not easily portable, so we just inform)
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start API: python api.py")
    print("2. Test API: python tests/test_api.py")
    print("3. View docs: http://localhost:8000/docs")
    print("4. View MLflow: mlflow ui")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
