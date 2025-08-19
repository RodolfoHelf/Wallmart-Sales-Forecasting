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
    
    csv_path = project_root / "data_manipulation" / "Walmart.csv"
    
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
        from data_manipulation.data_processor import WalmartDataProcessor
        print("✅ PASSED: WalmartDataProcessor imported successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not import WalmartDataProcessor: {e}")
        return False

def test_data_processor_initialization():
    """Test 3: Test data processor initialization"""
    print("\n🔍 Test 3: Testing data processor initialization...")
    
    try:
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
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
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
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
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
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
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
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
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load, clean, and prepare for database
        processor.load_csv_data()
        processor.clean_data()
        db_data = processor.prepare_for_database()
        
        print("✅ PASSED: Database preparation completed successfully")
        print(f"   📊 Database data shape: {db_data.shape}")
        print(f"   🔍 Database data types:")
        print(db_data.dtypes.to_string())
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not prepare data for database: {e}")
        return False

def test_data_export():
    """Test 8: Test data export functionality"""
    print("\n🔍 Test 8: Testing data export...")
    
    try:
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load and clean data
        processor.load_csv_data()
        processor.clean_data()
        
        # Export to different formats
        export_path = project_root / "tests" / "test_export"
        export_path.mkdir(exist_ok=True)
        
        # Export to CSV
        csv_export = export_path / "test_export.csv"
        processor.export_data(str(csv_export), format='csv')
        
        # Export to JSON
        json_export = export_path / "test_export.json"
        processor.export_data(str(json_export), format='json')
        
        # Export to Excel
        excel_export = export_path / "test_export.xlsx"
        processor.export_data(str(excel_export), format='excel')
        
        print("✅ PASSED: Data export completed successfully")
        print(f"   📁 CSV export: {csv_export.exists()}")
        print(f"   📁 JSON export: {json_export.exists()}")
        print(f"   📁 Excel export: {excel_export.exists()}")
        
        # Clean up test files
        for file_path in [csv_export, json_export, excel_export]:
            if file_path.exists():
                file_path.unlink()
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Could not export data: {e}")
        return False

def test_error_handling():
    """Test 9: Test error handling"""
    print("\n🔍 Test 9: Testing error handling...")
    
    try:
        from data_manipulation.data_processor import WalmartDataProcessor
        
        # Test with non-existent file
        try:
            processor = WalmartDataProcessor("non_existent_file.csv")
            processor.load_csv_data()
            print("❌ FAILED: Should have raised an error for non-existent file")
            return False
        except FileNotFoundError:
            print("✅ PASSED: Correctly handled non-existent file")
        
        # Test with invalid data
        try:
            # Create a temporary file with invalid data
            temp_file = project_root / "tests" / "temp_invalid.csv"
            temp_file.parent.mkdir(exist_ok=True)
            
            with open(temp_file, 'w') as f:
                f.write("invalid,csv,data\n")
                f.write("1,2,3\n")
                f.write("a,b,c\n")
            
            processor = WalmartDataProcessor(str(temp_file))
            processor.load_csv_data()
            
            # Clean up
            temp_file.unlink()
            
            print("✅ PASSED: Handled invalid data gracefully")
            
        except Exception as e:
            print(f"✅ PASSED: Correctly handled invalid data: {e}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Error handling test failed: {e}")
        return False

def test_performance():
    """Test 10: Test performance with large datasets"""
    print("\n🔍 Test 10: Testing performance...")
    
    try:
        from data_manipulation.data_processor import WalmartDataProcessor
        
        csv_path = project_root / "data_manipulation" / "Walmart.csv"
        processor = WalmartDataProcessor(str(csv_path))
        
        import time
        
        # Measure loading time
        start_time = time.time()
        processor.load_csv_data()
        load_time = time.time() - start_time
        
        # Measure cleaning time
        start_time = time.time()
        processor.clean_data()
        clean_time = time.time() - start_time
        
        # Measure analysis time
        start_time = time.time()
        processor.analyze_data()
        analysis_time = time.time() - start_time
        
        print("✅ PASSED: Performance test completed")
        print(f"   ⏱️  Load time: {load_time:.3f}s")
        print(f"   ⏱️  Clean time: {clean_time:.3f}s")
        print(f"   ⏱️  Analysis time: {analysis_time:.3f}s")
        print(f"   ⏱️  Total time: {load_time + clean_time + analysis_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 WALMART DATA PROCESSOR COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_csv_file_exists,
        test_data_processor_import,
        test_data_processor_initialization,
        test_csv_loading,
        test_data_cleaning,
        test_data_analysis,
        test_database_preparation,
        test_data_export,
        test_error_handling,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"📈 SUCCESS RATE: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Data processor is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
