#!/usr/bin/env python3
"""
Test script to verify CSV integration with actual Walmart.csv structure
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_csv_structure():
    """Test the actual CSV structure and basic operations"""
    print("🧪 Testing CSV Structure and Integration")
    print("=" * 50)
    
    try:
        # Test 1: Check if Walmart.csv exists
        print("1️⃣ Checking Walmart.csv...")
        csv_path = project_root / "data" / "Walmart.csv"
        
        if not csv_path.exists():
            print("   ❌ Walmart.csv not found!")
            return False
        
        print("   ✅ Walmart.csv found")
        
        # Test 2: Load and examine CSV structure
        print("2️⃣ Loading CSV and examining structure...")
        df = pd.read_csv(csv_path)
        
        print(f"   📊 Data shape: {df.shape}")
        print(f"   📋 Columns: {list(df.columns)}")
        print(f"   📅 Date column type: {df['Date'].dtype}")
        print(f"   📅 Sample dates: {df['Date'].head(3).tolist()}")
        
        # Test 3: Check for required columns
        print("3️⃣ Checking required columns...")
        required_columns = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"   ❌ Missing columns: {missing_columns}")
            return False
        else:
            print("   ✅ All required columns present")
        
        # Test 4: Test data processor import
        print("4️⃣ Testing data processor import...")
        try:
            from data.data_processor import WalmartDataProcessor
            print("   ✅ Data processor imported successfully")
        except Exception as e:
            print(f"   ❌ Data processor import failed: {e}")
            return False
        
        # Test 5: Test feature engineering import
        print("5️⃣ Testing feature engineering import...")
        try:
            from data.feature_engineering import WalmartFeatureEngineer
            print("   ✅ Feature engineering imported successfully")
        except Exception as e:
            print(f"   ❌ Feature engineering import failed: {e}")
            return False
        
        # Test 6: Test basic data processing
        print("6️⃣ Testing basic data processing...")
        try:
            processor = WalmartDataProcessor(str(csv_path))
            raw_data = processor.load_csv_data()
            print(f"   ✅ Data loaded: {raw_data.shape}")
            
            # Test Date column conversion
            processed_data = processor.clean_data()
            print(f"   ✅ Data cleaned: {processed_data.shape}")
            print(f"   📅 Date column type after processing: {processed_data['Date'].dtype}")
            
            if processed_data['Date'].dtype == 'datetime64[ns]':
                print("   ✅ Date column successfully converted to datetime")
            else:
                print(f"   ❌ Date column type is {processed_data['Date'].dtype}")
                return False
                
        except Exception as e:
            print(f"   ❌ Data processing failed: {e}")
            return False
        
        # Test 7: Test basic feature engineering
        print("7️⃣ Testing basic feature engineering...")
        try:
            # Take a small sample for testing
            sample_data = processed_data.head(100).copy()
            
            engineer = WalmartFeatureEngineer(sample_data)
            enhanced_data = engineer.create_temporal_features()
            
            print(f"   ✅ Temporal features created: {enhanced_data.shape}")
            print(f"   📈 New temporal columns: {[col for col in enhanced_data.columns if 'Date_' in col][:5]}...")
            
        except Exception as e:
            print(f"   ❌ Feature engineering failed: {e}")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 CSV INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Original CSV shape: {df.shape}")
        print(f"📊 Processed data shape: {processed_data.shape}")
        print(f"🚀 Enhanced data shape: {enhanced_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CSV integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing CSV Integration with Walmart Data")
    print("=" * 60)
    
    success = test_csv_structure()
    
    if success:
        print("\n🎉 CSV integration test passed! Ready for feature engineering.")
        sys.exit(0)
    else:
        print("\n❌ CSV integration test failed. Check the output above.")
        sys.exit(1)
