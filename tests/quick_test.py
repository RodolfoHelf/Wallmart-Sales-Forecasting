#!/usr/bin/env python3
"""
Quick test script for Walmart Data Processor
Simple test to check basic functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path (go up one level from tests directory)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def quick_test():
    """Quick test of the data processor"""
    print("🚀 Quick Test of Walmart Data Processor")
    print("=" * 50)
    
    try:
        # Test 1: Import
        print("1️⃣ Testing import...")
        from data_manipulation.data_processor import WalmartDataProcessor
        print("   ✅ Import successful")
        
        # Test 2: Check CSV file
        print("2️⃣ Checking CSV file...")
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
        if csv_path.exists():
            print(f"   ✅ CSV file found: {csv_path}")
            print(f"   📊 File size: {csv_path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"   ❌ CSV file not found: {csv_path}")
            return False
        
        # Test 3: Initialize processor
        print("3️⃣ Initializing processor...")
        processor = WalmartDataProcessor(str(csv_path))
        print("   ✅ Processor initialized")
        print(f"   📁 CSV path: {processor.csv_file_path}")
        
        # Test 4: Load data
        print("4️⃣ Loading CSV data...")
        data = processor.load_csv_data()
        print(f"   ✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"   📋 Column names: {list(data.columns)}")
        
        # Test 5: Show sample data
        print("5️⃣ Sample data (first 3 rows):")
        print(data.head(3).to_string())
        
        # Test 6: Basic data info
        print("6️⃣ Data types:")
        print(data.dtypes.to_string())
        
        # Test 7: Missing values check
        print("7️⃣ Missing values per column:")
        missing_counts = data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                print(f"   ⚠️  {col}: {count} missing values")
            else:
                print(f"   ✅ {col}: No missing values")
        
        print("\n🎉 Quick test completed successfully!")
        print(f"📊 Summary: {data.shape[0]} records, {data.shape[1]} columns")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
