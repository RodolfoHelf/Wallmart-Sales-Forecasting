#!/usr/bin/env python3
"""
Test script to verify Date column conversion in data_processor
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_date_conversion():
    """Test the Date column conversion functionality"""
    print("🧪 Testing Date Column Conversion in Data Processor")
    print("=" * 60)
    
    try:
        # Test 1: Import the module
        print("1️⃣ Testing import...")
        from data.data_processor import WalmartDataProcessor
        print("   ✅ Data processor module imported successfully")
        
        # Test 2: Check if Walmart.csv exists
        print("2️⃣ Checking data file...")
        csv_path = project_root / "data" / "Walmart.csv"
        
        if not csv_path.exists():
            print("   ❌ Walmart.csv not found. Creating sample data...")
            # Create sample data with Date column as object type
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            sample_data = pd.DataFrame({
                'Store': np.random.randint(1, 6, 100),
                'Dept': np.random.randint(1, 4, 100),
                'Date': dates.strftime('%Y-%m-%d'),  # This creates object type
                'Weekly_Sales': np.random.uniform(10000, 50000, 100),
                'IsHoliday': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
                'Temperature': np.random.uniform(20, 80, 100),
                'Fuel_Price': np.random.uniform(2.5, 4.0, 100),
                'CPI': np.random.uniform(200, 250, 100),
                'Unemployment': np.random.uniform(3.0, 8.0, 100)
            })
            
            # Save sample data
            sample_data.to_csv(csv_path, index=False)
            print(f"   ✅ Created sample data with {len(sample_data)} records")
        else:
            print("   ✅ Walmart.csv found")
        
        # Test 3: Initialize data processor
        print("3️⃣ Initializing data processor...")
        processor = WalmartDataProcessor(str(csv_path))
        print("   ✅ Data processor initialized")
        
        # Test 4: Load data and check initial types
        print("4️⃣ Loading data and checking initial types...")
        data = processor.load_csv_data()
        print(f"   📊 Data shape: {data.shape}")
        print(f"   📋 Columns: {list(data.columns)}")
        print(f"   📅 Date column type before processing: {data['Date'].dtype}")
        print(f"   📅 Date column sample values: {data['Date'].head(3).tolist()}")
        
        # Test 5: Verify Date column conversion
        print("5️⃣ Verifying Date column conversion...")
        date_ok = processor.verify_date_conversion()
        if date_ok:
            print("   ✅ Date column conversion verified successfully")
        else:
            print("   ⚠️  Date column conversion issues detected")
        
        # Test 6: Clean data (this should convert Date column)
        print("6️⃣ Cleaning data (should convert Date column)...")
        clean_data = processor.clean_data()
        print(f"   📊 Clean data shape: {clean_data.shape}")
        print(f"   📅 Date column type after cleaning: {clean_data['Date'].dtype}")
        
        # Test 7: Final verification
        print("7️⃣ Final verification of Date column...")
        if clean_data['Date'].dtype == 'datetime64[ns]':
            print("   ✅ Date column successfully converted to datetime")
            print(f"   📅 Date range: {clean_data['Date'].min()} to {clean_data['Date'].max()}")
            print(f"   📅 Total unique dates: {clean_data['Date'].nunique()}")
        else:
            print(f"   ❌ Date column type is still {clean_data['Date'].dtype}")
        
        # Test 8: Analyze data to see Date column analysis
        print("8️⃣ Analyzing data to see Date column analysis...")
        analysis = processor.analyze_data()
        
        if 'date_column_analysis' in analysis:
            date_analysis = analysis['date_column_analysis']
            print("   📊 Date Column Analysis:")
            print(f"      - Column name: {date_analysis['column_name']}")
            print(f"      - Data type: {date_analysis['data_type']}")
            print(f"      - Is datetime: {date_analysis['is_datetime']}")
            print(f"      - Total records: {date_analysis['total_records']}")
            print(f"      - Unique dates: {date_analysis['unique_dates']}")
            print(f"      - Missing values: {date_analysis['missing_values']}")
            print(f"      - Sample values: {date_analysis['sample_values']}")
        else:
            print("   ⚠️  No date column analysis found")
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 DATE CONVERSION TEST COMPLETED!")
        print("=" * 60)
        
        if clean_data['Date'].dtype == 'datetime64[ns]':
            print("✅ SUCCESS: Date column is properly converted to datetime")
            print(f"📊 Data shape: {clean_data.shape}")
            print(f"📅 Date range: {clean_data['Date'].min()} to {clean_data['Date'].max()}")
        else:
            print("❌ FAILED: Date column is not converted to datetime")
            print(f"📅 Current type: {clean_data['Date'].dtype}")
        
        return clean_data['Date'].dtype == 'datetime64[ns]'
        
    except Exception as e:
        print(f"\n❌ Date conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_conversion():
    """Test manual Date column conversion"""
    print("\n🔧 Testing Manual Date Column Conversion")
    print("=" * 40)
    
    try:
        # Create sample data with Date as object
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'Store': np.random.randint(1, 4, 50),
            'Dept': np.random.randint(1, 3, 50),
            'Date': dates.strftime('%Y-%m-%d'),  # Object type
            'Weekly_Sales': np.random.uniform(10000, 50000, 50)
        })
        
        print(f"📊 Sample data shape: {sample_data.shape}")
        print(f"📅 Date column type: {sample_data['Date'].dtype}")
        print(f"📅 Sample dates: {sample_data['Date'].head(3).tolist()}")
        
        # Convert Date column manually
        print("\n🔄 Converting Date column to datetime...")
        sample_data['Date'] = pd.to_datetime(sample_data['Date'], errors='coerce')
        
        print(f"📅 Date column type after conversion: {sample_data['Date'].dtype}")
        print(f"📅 Sample dates after conversion: {sample_data['Date'].head(3).tolist()}")
        
        # Check for any conversion errors
        invalid_dates = sample_data['Date'].isna().sum()
        if invalid_dates > 0:
            print(f"⚠️  Found {invalid_dates} invalid dates after conversion")
        else:
            print("✅ All dates converted successfully")
        
        return sample_data['Date'].dtype == 'datetime64[ns]'
        
    except Exception as e:
        print(f"❌ Manual conversion test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Date Column Conversion in Walmart Data Processor")
    print("=" * 70)
    
    # Run main test
    success1 = test_date_conversion()
    
    # Run manual conversion test
    success2 = test_manual_conversion()
    
    # Final result
    if success1 and success2:
        print("\n🎉 All tests passed! Date column conversion is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)
