#!/usr/bin/env python3
"""
Test script for Walmart Feature Engineering Module
Demonstrates how to use the feature engineering functionality
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_feature_engineering():
    """Test the feature engineering module with data processor"""
    print("🚀 Testing Walmart Feature Engineering Module with Data Processor")
    print("=" * 70)
    
    try:
        # Test 1: Import the modules
        print("1️⃣ Testing imports...")
        from data.feature_engineering import WalmartFeatureEngineer
        from data.data_processor import WalmartDataProcessor
        print("   ✅ Feature engineering and data processor modules imported successfully")
        
        # Test 2: Load and process data using data processor
        print("2️⃣ Loading and processing data with data processor...")
        csv_path = project_root / "data" / "Walmart.csv"
        
        if not csv_path.exists():
            print("   ❌ Walmart.csv not found. Creating sample data...")
            # Create sample data with proper column names for data processor
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            sample_data = pd.DataFrame({
                'Store': np.random.randint(1, 6, 100),
                'Date': dates.strftime('%Y-%m-%d'),  # This will be object type initially
                'Weekly_Sales': np.random.uniform(10000, 50000, 100),
                'Holiday_Flag': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
                'Temperature': np.random.uniform(20, 80, 100),
                'Fuel_Price': np.random.uniform(2.5, 4.0, 100),
                'CPI': np.random.uniform(200, 250, 100),
                'Unemployment': np.random.uniform(3.0, 8.0, 100)
            })
            
            # Save sample data
            sample_data.to_csv(csv_path, index=False)
            print(f"   ✅ Created sample data with {len(sample_data)} records")
        else:
            print("   ✅ Walmart.csv found, using sample for testing...")
            # Load the actual CSV and take a sample for testing
            sample_data = pd.read_csv(csv_path)
            # Take a sample of 1000 records for testing to avoid memory issues
            if len(sample_data) > 1000:
                sample_data = sample_data.sample(n=1000, random_state=42).reset_index(drop=True)
                print(f"   📊 Sampled {len(sample_data)} records from {len(pd.read_csv(csv_path))} total records")
            else:
                print(f"   📊 Using all {len(sample_data)} records")
            
            # Save the sample for consistent testing
            sample_csv_path = project_root / "data" / "Walmart_sample.csv"
            sample_data.to_csv(sample_csv_path, index=False)
            print(f"   💾 Saved sample to {sample_csv_path.name}")
            csv_path = sample_csv_path  # Use the sample for processing
        
        # Test 3: Process data using data processor
        print("3️⃣ Processing data with data processor...")
        processor = WalmartDataProcessor(str(csv_path))
        
        # Load data
        raw_data = processor.load_csv_data()
        print(f"   📊 Raw data shape: {raw_data.shape}")
        print(f"   📅 Date column type before processing: {raw_data['Date'].dtype}")
        
        # Clean data (this converts Date column to datetime)
        processed_data = processor.clean_data()
        print(f"   📊 Processed data shape: {processed_data.shape}")
        print(f"   📅 Date column type after processing: {processed_data['Date'].dtype}")
        
        # Verify Date column conversion
        if processed_data['Date'].dtype == 'datetime64[ns]':
            print("   ✅ Date column successfully converted to datetime by data processor")
        else:
            print(f"   ❌ Date column type is still {processed_data['Date'].dtype}")
            return False
        
        # Test 4: Initialize feature engineer with processed data
        print("4️⃣ Initializing feature engineer with processed data...")
        engineer = WalmartFeatureEngineer(processed_data)
        print("   ✅ Feature engineer initialized")
        
        # Test 5: Create temporal features
        print("5️⃣ Creating temporal features...")
        enhanced_data = engineer.create_temporal_features()
        print(f"   ✅ Temporal features created. New shape: {enhanced_data.shape}")
        print(f"   📈 New temporal columns: {[col for col in enhanced_data.columns if 'Date_' in col][:5]}...")
        
        # Test 6: Create lag features
        print("6️⃣ Creating lag features...")
        enhanced_data = engineer.create_lag_features('Weekly_Sales', lags=[1, 2, 3, 7])
        print(f"   ✅ Lag features created. New shape: {enhanced_data.shape}")
        print(f"   📈 New lag columns: {[col for col in enhanced_data.columns if 'lag_' in col]}")
        
        # Test 7: Create rolling features
        print("7️⃣ Creating rolling features...")
        enhanced_data = engineer.create_rolling_features('Weekly_Sales', windows=[3, 7])
        print(f"   ✅ Rolling features created. New shape: {enhanced_data.shape}")
        print(f"   📈 New rolling columns: {[col for col in enhanced_data.columns if 'rolling_' in col][:5]}...")
        
        # Test 8: Create holiday features
        print("8️⃣ Creating holiday features...")
        enhanced_data = engineer.create_holiday_features('Date')
        print(f"   ✅ Holiday features created. New shape: {enhanced_data.shape}")
        print(f"   🎉 New holiday columns: {[col for col in enhanced_data.columns if 'is_' in col]}")
        
        # Test 9: Create encoding features
        print("9️⃣ Creating encoding features...")
        categorical_cols = ['Store']  # Only Store column exists
        enhanced_data = engineer.create_encoding_features(categorical_cols)
        print(f"   ✅ Encoding features created. New shape: {enhanced_data.shape}")
        print(f"   🏷️  New encoding columns: {[col for col in enhanced_data.columns if '_encoded' in col or '_freq' in col]}")
        
        # Test 10: Create interaction features
        print("🔟 Creating interaction features...")
        numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price']
        enhanced_data = engineer.create_interaction_features(numeric_cols)
        print(f"   ✅ Interaction features created. New shape: {enhanced_data.shape}")
        print(f"   🔗 New interaction columns: {[col for col in enhanced_data.columns if '_x_' in col or '_plus_' in col][:5]}...")
        
        # Test 11: Create statistical features
        print("1️⃣1️⃣ Creating statistical features...")
        enhanced_data = engineer.create_statistical_features('Weekly_Sales', ['Store'])  # Only Store column exists
        print(f"   ✅ Statistical features created. New shape: {enhanced_data.shape}")
        
        # Test 12: Create weather features
        print("1️⃣2️⃣ Creating weather features...")
        enhanced_data = engineer.create_weather_features('Temperature', 'Fuel_Price')
        print(f"   ✅ Weather features created. New shape: {enhanced_data.shape}")
        
        # Test 13: Get feature summary
        print("1️⃣3️⃣ Getting feature summary...")
        summary = engineer.get_feature_summary()
        print(f"   ✅ Feature summary created")
        print(f"   📊 Total features created: {summary['total_features']}")
        print(f"   📋 Original columns: {summary['original_columns']}")
        
        # Test 14: Remove correlated features
        print("1️⃣4️⃣ Removing correlated features...")
        enhanced_data = engineer.remove_correlated_features(threshold=0.95)
        print(f"   ✅ Correlated features removed. Final shape: {enhanced_data.shape}")
        
        # Test 15: Scale features
        print("1️⃣5️⃣ Scaling features...")
        enhanced_data = engineer.scale_features(method='standard')
        print(f"   ✅ Features scaled. Final shape: {enhanced_data.shape}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 FEATURE ENGINEERING TEST WITH DATA PROCESSOR COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Raw data shape: {raw_data.shape}")
        print(f"📊 Processed data shape: {processed_data.shape}")
        print(f"🚀 Enhanced data shape: {enhanced_data.shape}")
        print(f"✨ Total new features created: {summary['total_features']}")
        print(f"📈 Feature columns: {len(engineer.feature_columns)}")
        
        # Show sample of new features
        new_features = [col for col in enhanced_data.columns if col not in processed_data.columns]
        print(f"\n🔍 Sample of new features created:")
        for i, feature in enumerate(new_features[:10]):
            print(f"   {i+1:2d}. {feature}")
        if len(new_features) > 10:
            print(f"   ... and {len(new_features) - 10} more features")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_demo():
    """Quick demonstration of feature engineering with data processor"""
    print("\n🚀 Quick Feature Engineering Demo with Data Processor")
    print("=" * 55)
    
    try:
        from data.feature_engineering import WalmartFeatureEngineer
        from data.data_processor import WalmartDataProcessor
        
        # Create simple sample data with proper column names
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        demo_data = pd.DataFrame({
            'Store': np.random.randint(1, 4, 50),
            'Date': dates.strftime('%Y-%m-%d'),  # Object type initially
            'Weekly_Sales': np.random.uniform(10000, 50000, 50),
            'Temperature': np.random.uniform(20, 80, 50),
            'Fuel_Price': np.random.uniform(2.5, 4.0, 50),
            'Holiday_Flag': np.random.choice([0, 1], 50, p=[0.8, 0.2])
        })
        
        print(f"📊 Demo data shape: {demo_data.shape}")
        print(f"📅 Date column type before processing: {demo_data['Date'].dtype}")
        
        # Process data using data processor
        print("🔄 Processing data with data processor...")
        processor = WalmartDataProcessor("temp_demo.csv")
        processor.data = demo_data  # Set data directly
        
        # Clean data (convert Date column to datetime)
        processed_data = processor.clean_data()
        print(f"📅 Date column type after processing: {processed_data['Date'].dtype}")
        
        # Initialize feature engineer with processed data
        engineer = WalmartFeatureEngineer(processed_data)
        
        # Create all features at once
        enhanced_data = engineer.create_all_features(
            target_col='Weekly_Sales',
            date_col='Date',
            categorical_cols=['Store'],  # Only Store column exists
            numeric_cols=['Temperature', 'Fuel_Price']
        )
        
        print(f"🚀 Enhanced data shape: {enhanced_data.shape}")
        print(f"✨ Features created: {len(engineer.feature_columns)}")
        
        # Show feature summary
        summary = engineer.get_feature_summary()
        print(f"📈 Total features: {summary['total_features']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Walmart Feature Engineering Module")
    print("=" * 60)
    
    # Run comprehensive test
    success1 = test_feature_engineering()
    
    # Run quick demo
    success2 = quick_demo()
    
    # Final result
    if success1 and success2:
        print("\n🎉 All tests passed! Feature engineering module is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)
