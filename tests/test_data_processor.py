#!/usr/bin/env python3
"""
Comprehensive test script for Walmart Data Processor
This script tests all the main functionality of the data processor
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the project root to the path (go up one level from tests directory)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_csv_file_exists():
    """Test 1: Check if Walmart.csv exists and is readable"""
    print("🔍 Test 1: Checking if Walmart.csv exists...")
    
    csv_path = project_root / "data" / "Walmart.csv"
    
    if not csv_path.exists():
        print("❌ FAILED: Walmart.csv not found!")
        return False
    
    try:
        # Try to read the file
        df = pd.read_csv(csv_path)
        print(f"✅ PASSED: Walmart.csv found and readable")
        print(f"   📊 File size: {csv_path.stat().st_size / 1024:.1f} KB")
        print(f"   📈 Records: {len(df)}")
        print(f"   🗂️  Columns: {len(df.columns)}")
        print(f"   📋 Column names: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not read CSV file: {e}")
        return False

def test_data_processor_import():
    """Test 2: Check if data processor can be imported"""
    print("\n🔍 Test 2: Testing data processor import...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        print("✅ PASSED: WalmartDataProcessor imported successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not import WalmartDataProcessor: {e}")
        return False

def test_data_processor_initialization():
    """Test 3: Test data processor initialization"""
    print("\n🔍 Test 3: Testing data processor initialization...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        print("✅ PASSED: Data processor initialized successfully")
        print(f"   📁 CSV path: {processor.csv_file_path}")
        print(f"   📊 Data attribute: {processor.data is None}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not initialize data processor: {e}")
        return False

def test_csv_loading():
    """Test 4: Test CSV loading functionality"""
    print("\n🔍 Test 4: Testing CSV loading...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load the data
        data = processor.load_csv_data()
        
        print("✅ PASSED: CSV data loaded successfully")
        print(f"   📊 Data shape: {data.shape}")
        print(f"   📋 Data types: {data.dtypes.to_dict()}")
        print(f"   🔍 First few rows:")
        print(data.head(3).to_string())
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not load CSV data: {e}")
        return False

def test_data_cleaning():
    """Test 5: Test data cleaning functionality"""
    print("\n🔍 Test 5: Testing data cleaning...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load and clean data
        processor.load_csv_data()
        clean_data = processor.clean_data()
        
        print("✅ PASSED: Data cleaning completed successfully")
        print(f"   📊 Original shape: {processor.data.shape}")
        print(f"   🧹 Cleaned shape: {clean_data.shape}")
        print(f"   🔍 Missing values after cleaning:")
        print(clean_data.isnull().sum().to_string())
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not clean data: {e}")
        return False

def test_data_analysis():
    """Test 6: Test data analysis functionality"""
    print("\n🔍 Test 6: Testing data analysis...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load, clean, and analyze data
        processor.load_csv_data()
        processor.clean_data()
        analysis = processor.analyze_data()
        
        print("✅ PASSED: Data analysis completed successfully")
        print(f"   📊 Basic stats: {analysis['basic_stats']}")
        print(f"   🗂️  Column types: {analysis['columns']['types']}")
        
        if 'numeric_analysis' in analysis:
            print(f"   📈 Numeric columns: {analysis['numeric_analysis']['columns']}")
        
        if 'categorical_analysis' in analysis:
            print(f"   🏷️  Categorical columns: {list(analysis['categorical_analysis'].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not analyze data: {e}")
        return False

def test_database_preparation():
    """Test 7: Test database preparation functionality"""
    print("\n🔍 Test 7: Testing database preparation...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load, clean, and prepare for database
        processor.load_csv_data()
        processor.clean_data()
        records = processor.prepare_for_database()
        
        print("✅ PASSED: Database preparation completed successfully")
        print(f"   📊 Records prepared: {len(records)}")
        if records:
            print(f"   🗂️  Sample record keys: {list(records[0].keys())}")
            print(f"   🔍 Sample record: {records[0]}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not prepare data for database: {e}")
        return False

def test_sample_data_processing():
    """Test 8: Test processing with a small sample"""
    print("\n🔍 Test 8: Testing sample data processing...")
    
    try:
        from data.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load data
        data = processor.load_csv_data()
        
        # Take a small sample for testing
        sample_data = data.head(100).copy()
        processor.data = sample_data
        
        # Test cleaning on sample
        clean_sample = processor.clean_data()
        print(f"✅ PASSED: Sample cleaning completed")
        print(f"   📊 Sample size: {len(clean_sample)}")
        
        # Test analysis on sample
        sample_analysis = processor.analyze_data()
        print(f"✅ PASSED: Sample analysis completed")
        print(f"   📊 Sample stats: {sample_analysis['basic_stats']}")
        
        # Test database preparation on sample
        sample_records = processor.prepare_for_database()
        print(f"✅ PASSED: Sample database preparation completed")
        print(f"   📊 Sample records: {len(sample_records)}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Sample processing failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting Walmart Data Processor Tests...")
    print("=" * 60)
    
    tests = [
        ("CSV File Exists", test_csv_file_exists),
        ("Data Processor Import", test_data_processor_import),
        ("Data Processor Initialization", test_data_processor_initialization),
        ("CSV Loading", test_csv_loading),
        ("Data Cleaning", test_data_cleaning),
        ("Data Analysis", test_data_analysis),
        ("Database Preparation", test_database_preparation),
        ("Sample Data Processing", test_sample_data_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data processor is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
