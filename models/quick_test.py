#!/usr/bin/env python3
"""
Quick test for Walmart sales forecasting models - Fast execution version
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickModelTester:
    """Quick tester for Walmart sales forecasting models"""
    
    def __init__(self):
        """Initialize the quick model tester"""
        self.test_results = {}
        
    def generate_test_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic test data for quick testing"""
        logger.info(f"Generating {n_samples} synthetic data points...")
        
        # Generate dates (weekly intervals)
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(weeks=i) for i in range(n_samples)]
        
        # Generate synthetic sales data with seasonality and trend
        np.random.seed(42)  # For reproducible results
        
        # Base sales with trend
        base_sales = 50000 + np.arange(n_samples) * 100  # Upward trend
        
        # Add seasonality (weekly and monthly patterns)
        weekly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 4) * 5000  # 4-week cycle
        monthly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 13) * 8000  # 13-week cycle
        
        # Add holiday effects (every 13 weeks)
        holiday_effect = np.where(np.arange(n_samples) % 13 == 0, 15000, 0)
        
        # Add noise
        noise = np.random.normal(0, 3000, n_samples)
        
        # Combine all effects
        weekly_sales = base_sales + weekly_pattern + monthly_pattern + holiday_effect + noise
        weekly_sales = np.maximum(weekly_sales, 1000)  # Ensure positive sales
        
        # Generate other features
        temperature = 20 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 52) + np.random.normal(0, 5, n_samples)
        fuel_price = 2.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 26) + np.random.normal(0, 0.1, n_samples)
        cpi = 100 + np.arange(n_samples) * 0.1 + np.random.normal(0, 0.5, n_samples)
        unemployment_rate = 5 + 2 * np.sin(2 * np.pi * np.arange(n_samples) / 52) + np.random.normal(0, 0.3, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'weekly_sales': weekly_sales,
            'temperature': temperature,
            'fuel_price': fuel_price,
            'cpi': cpi,
            'unemployment_rate': unemployment_rate,
            'is_holiday': np.arange(n_samples) % 13 == 0
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        logger.info(f"Generated test data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def quick_sarimax_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Quick SARIMAX-like test (simplified)"""
        start_time = time.time()
        
        try:
            # Simple trend analysis
            sales_series = data['weekly_sales'].values
            x = np.arange(len(sales_series))
            
            # Linear trend
            trend_coef = np.polyfit(x, sales_series, 1)[0]
            
            # Simple seasonality detection
            seasonal_pattern = np.array([sales_series[i::4].mean() for i in range(4)])
            
            # Calculate basic metrics
            mean_sales = np.mean(sales_series)
            std_sales = np.std(sales_series)
            mape = np.mean(np.abs((sales_series - mean_sales) / sales_series)) * 100
            
            # Simple forecast (next 4 weeks)
            last_value = sales_series[-1]
            forecast = [last_value + trend_coef * (i + 1) for i in range(4)]
            
            execution_time = time.time() - start_time
            
            result = {
                'model_name': 'quick_sarimax',
                'execution_time': execution_time,
                'mape_score': mape,
                'trend_coefficient': trend_coef,
                'mean_sales': mean_sales,
                'std_sales': std_sales,
                'forecast_next_4_weeks': forecast,
                'status': 'success'
            }
            
            logger.info(f"Quick SARIMAX test completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quick SARIMAX test failed: {str(e)}")
            return {
                'model_name': 'quick_sarimax',
                'execution_time': execution_time,
                'error': str(e),
                'status': 'failed'
            }
    
    def quick_lightgbm_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Quick LightGBM-like test (simplified)"""
        start_time = time.time()
        
        try:
            # Feature engineering
            data_clean = data.copy()
            data_clean['day_of_week'] = data_clean.index.dayofweek
            data_clean['month'] = data_clean.index.month
            data_clean['quarter'] = data_clean.index.quarter
            data_clean['lag_1'] = data_clean['weekly_sales'].shift(1)
            data_clean['lag_4'] = data_clean['weekly_sales'].shift(4)
            data_clean['rolling_mean_4'] = data_clean['weekly_sales'].rolling(4).mean()
            
            # Remove NaN values
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 20:
                raise ValueError("Insufficient data after feature engineering")
            
            # Simple feature importance (correlation with target)
            feature_cols = ['day_of_week', 'month', 'quarter', 'lag_1', 'lag_4', 'rolling_mean_4']
            correlations = {}
            
            for col in feature_cols:
                if col in data_clean.columns:
                    corr = np.corrcoef(data_clean[col], data_clean['weekly_sales'])[0, 1]
                    correlations[col] = corr if not np.isnan(corr) else 0
            
            # Simple prediction using rolling average
            last_4_weeks = data_clean['weekly_sales'].tail(4).values
            forecast = np.mean(last_4_weeks) * 1.02  # 2% growth assumption
            
            # Calculate metrics
            mean_sales = np.mean(data_clean['weekly_sales'])
            mape = np.mean(np.abs((data_clean['weekly_sales'] - mean_sales) / data_clean['weekly_sales'])) * 100
            
            execution_time = time.time() - start_time
            
            result = {
                'model_name': 'quick_lightgbm',
                'execution_time': execution_time,
                'mape_score': mape,
                'mean_sales': mean_sales,
                'forecast_next_week': forecast,
                'feature_importance': correlations,
                'status': 'success'
            }
            
            logger.info(f"Quick LightGBM test completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quick LightGBM test failed: {str(e)}")
            return {
                'model_name': 'quick_lightgbm',
                'execution_time': execution_time,
                'error': str(e),
                'status': 'failed'
            }
    
    def quick_ensemble_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Quick ensemble test combining multiple approaches"""
        start_time = time.time()
        
        try:
            # Get individual model results
            sarimax_result = self.quick_sarimax_test(data)
            lightgbm_result = self.quick_lightgbm_test(data)
            
            # Simple ensemble (average of forecasts)
            if sarimax_result['status'] == 'success' and lightgbm_result['status'] == 'success':
                sarimax_forecast = sarimax_result['forecast_next_4_weeks'][0]  # First week
                lightgbm_forecast = lightgbm_result['forecast_next_week']
                
                ensemble_forecast = (sarimax_forecast + lightgbm_forecast) / 2
                
                # Calculate ensemble MAPE (average of individual MAPEs)
                ensemble_mape = (sarimax_result['mape_score'] + lightgbm_result['mape_score']) / 2
                
                result = {
                    'model_name': 'quick_ensemble',
                    'execution_time': time.time() - start_time,
                    'ensemble_mape': ensemble_mape,
                    'sarimax_forecast': sarimax_forecast,
                    'lightgbm_forecast': lightgbm_forecast,
                    'ensemble_forecast': ensemble_forecast,
                    'individual_results': {
                        'sarimax': sarimax_result,
                        'lightgbm': lightgbm_result
                    },
                    'status': 'success'
                }
            else:
                raise ValueError("Individual models failed, cannot create ensemble")
            
            logger.info(f"Quick ensemble test completed in {time.time() - start_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quick ensemble test failed: {str(e)}")
            return {
                'model_name': 'quick_ensemble',
                'execution_time': execution_time,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_quick_test_suite(self, n_samples: int = 100) -> Dict[str, Any]:
        """Run the complete quick test suite"""
        logger.info("🚀 Starting Quick Model Test Suite...")
        start_time = time.time()
        
        try:
            # Generate test data
            test_data = self.generate_test_data(n_samples)
            
            # Run individual model tests
            logger.info("📊 Testing Quick SARIMAX...")
            sarimax_result = self.quick_sarimax_test(test_data)
            
            logger.info("🌳 Testing Quick LightGBM...")
            lightgbm_result = self.quick_lightgbm_test(test_data)
            
            logger.info("🔗 Testing Quick Ensemble...")
            ensemble_result = self.quick_ensemble_test(test_data)
            
            # Compile results
            total_time = time.time() - start_time
            
            test_summary = {
                'test_timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'data_samples': n_samples,
                'models_tested': 3,
                'successful_models': sum(1 for r in [sarimax_result, lightgbm_result, ensemble_result] 
                                       if r['status'] == 'success'),
                'results': {
                    'sarimax': sarimax_result,
                    'lightgbm': lightgbm_result,
                    'ensemble': ensemble_result
                },
                'data_summary': {
                    'total_rows': len(test_data),
                    'date_range': f"{test_data.index.min().date()} to {test_data.index.max().date()}",
                    'mean_sales': test_data['weekly_sales'].mean(),
                    'sales_std': test_data['weekly_sales'].std()
                }
            }
            
            # Log summary
            logger.info("✅ Quick Test Suite Completed Successfully!")
            logger.info(f"⏱️  Total execution time: {total_time:.3f}s")
            logger.info(f"📈 Models tested: {test_summary['models_tested']}")
            logger.info(f"✅ Successful models: {test_summary['successful_models']}")
            
            return test_summary
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Quick Test Suite failed: {str(e)}")
            return {
                'test_timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'error': str(e),
                'status': 'failed'
            }

def main():
    """Main function to run the quick test"""
    print("🏪 Walmart Sales Forecasting - Quick Model Test")
    print("=" * 50)
    
    # Create tester instance
    tester = QuickModelTester()
    
    # Run quick test suite
    results = tester.run_quick_test_suite(n_samples=100)
    
    # Print results summary
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    
    if 'error' not in results:
        print(f"✅ Status: Success")
        print(f"⏱️  Total Time: {results['total_execution_time']:.3f}s")
        print(f"📊 Models Tested: {results['models_tested']}")
        print(f"✅ Successful: {results['successful_models']}")
        
        print(f"\n📈 Data Summary:")
        print(f"   • Samples: {results['data_summary']['total_rows']}")
        print(f"   • Date Range: {results['data_summary']['date_range']}")
        print(f"   • Mean Sales: ${results['data_summary']['mean_sales']:,.0f}")
        print(f"   • Sales Std: ${results['data_summary']['sales_std']:,.0f}")
        
        print(f"\n🔍 Individual Model Results:")
        for model_name, result in results['results'].items():
            if result['status'] == 'success':
                if result.get('mape_score') is not None:
                    print(f"   • {model_name.upper()}: {result['execution_time']:.3f}s, MAPE: {result['mape_score']:.2f}%")
                else:
                    print(f"   • {model_name.upper()}: {result['execution_time']:.3f}s, MAPE: N/A")
            else:
                print(f"   • {model_name.upper()}: FAILED - {result.get('error', 'Unknown error')}")
    else:
        print(f"❌ Status: Failed")
        print(f"❌ Error: {results['error']}")
        print(f"⏱️  Time before failure: {results['total_execution_time']:.3f}s")
    
    print("\n" + "=" * 50)
    print("🎯 Quick test completed! Check logs for detailed results.")

if __name__ == "__main__":
    main()
