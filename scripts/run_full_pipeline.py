#!/usr/bin/env python3
"""
Full Pipeline Script for Walmart Sales Forecasting
Runs data processing → feature engineering → model training
"""

import os
import sys

# Set up paths BEFORE any other imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Now import the required libraries
import logging
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import our modules
from data_manipulation import (
    WalmartDataProcessor,
    WalmartFeatureEngineer,
    cross_validate_model
)

# Try to import MLflow-dependent modules, but make them optional
try:
    from models.train_models import ModelTrainer
    MLFLOW_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Warning: MLflow modules not available: {e}")
    print("   Model training will be skipped. Continuing with data processing...")
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./outputs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WalmartPipelineRunner:
    """Runs the complete Walmart sales forecasting pipeline"""
    
    def __init__(self, input_csv_path: str = "data_manipulation/Walmart.csv"):
        """
        Initialize the pipeline runner
        
        Args:
            input_csv_path: Path to the input Walmart.csv file
        """
        self.input_csv_path = input_csv_path
        self.output_dir = Path("./outputs/pipeline_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Pipeline outputs
        self.processed_data_path = self.output_dir / "processed_data.csv"
        self.featured_data_path = self.output_dir / "featured_data.csv"
        self.models_dir = self.output_dir / "trained_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Pipeline results
        self.pipeline_results = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'stages': {},
            'data_info': {},
            'model_results': {},
            'mlflow_status': 'disabled' if not MLFLOW_AVAILABLE else 'enabled'
        }
    
    def run_full_pipeline(self) -> dict:
        """Run the complete pipeline from data processing to model training"""
        start_time = time.time()
        self.pipeline_results['start_time'] = datetime.now().isoformat()
        
        logger.info("🚀 Starting Walmart Sales Forecasting Full Pipeline")
        logger.info("=" * 60)
        
        if not MLFLOW_AVAILABLE:
            logger.warning("⚠️  MLflow not available - model training will be skipped")
            logger.info("   To enable model training, ensure MLflow server is running or fix dependencies")
        
        try:
            # Stage 1: Data Processing
            logger.info("📊 STAGE 1: Data Processing")
            logger.info("-" * 40)
            processed_data = self._run_data_processing()
            
            # Stage 2: Feature Engineering
            logger.info("🔧 STAGE 2: Feature Engineering")
            logger.info("-" * 40)
            featured_data = self._run_feature_engineering(processed_data)
            
            # Stage 3: Model Training (only if MLflow is available)
            if MLFLOW_AVAILABLE:
                logger.info("🤖 STAGE 3: Model Training")
                logger.info("-" * 40)
                model_results = self._run_model_training(featured_data)
            else:
                logger.info("⏭️  STAGE 3: Model Training (SKIPPED - MLflow not available)")
                logger.info("-" * 40)
                model_results = {'status': 'skipped', 'reason': 'MLflow not available'}
                self.pipeline_results['stages']['model_training'] = {
                    'status': 'skipped',
                    'reason': 'MLflow not available'
                }
            
            # Pipeline completion
            end_time = time.time()
            total_duration = end_time - start_time
            self.pipeline_results['end_time'] = datetime.now().isoformat()
            self.pipeline_results['total_duration'] = total_duration
            
            # Save pipeline summary
            self._save_pipeline_summary()
            
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            self.pipeline_results['error'] = str(e)
            self.pipeline_results['end_time'] = datetime.now().isoformat()
            self._save_pipeline_summary()
            raise
    
    def _run_data_processing(self) -> pd.DataFrame:
        """Run data processing stage"""
        stage_start = time.time()
        
        try:
            logger.info(f"📁 Input file: {self.input_csv_path}")
            
            # Check if input file exists
            if not os.path.exists(self.input_csv_path):
                raise FileNotFoundError(f"Input file not found: {self.input_csv_path}")
            
            # Initialize data processor
            processor = WalmartDataProcessor(self.input_csv_path)
            
            # Load and process data
            logger.info("Loading CSV data...")
            raw_data = processor.load_csv_data()
            
            logger.info("Cleaning and processing data...")
            processed_data = processor.clean_data()
            
            # Save processed data
            logger.info(f"Saving processed data to {self.processed_data_path}")
            processed_data.to_csv(self.processed_data_path, index=False)
            
            # Record stage results
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['data_processing'] = {
                'status': 'success',
                'duration': stage_duration,
                'input_rows': len(raw_data),
                'output_rows': len(processed_data),
                'input_columns': len(raw_data.columns),
                'output_columns': len(processed_data.columns),
                'output_file': str(self.processed_data_path)
            }
            
            # Data info
            self.pipeline_results['data_info']['processed'] = {
                'rows': len(processed_data),
                'columns': len(processed_data.columns),
                'columns_list': list(processed_data.columns),
                'date_range': f"{processed_data['Date'].min()} to {processed_data['Date'].max()}" if 'Date' in processed_data.columns else "N/A",
                'memory_usage_mb': processed_data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"✅ Data processing completed in {stage_duration:.2f}s")
            logger.info(f"   • Input: {len(raw_data)} rows, {len(raw_data.columns)} columns")
            logger.info(f"   • Output: {len(processed_data)} rows, {len(processed_data.columns)} columns")
            logger.info(f"   • Saved to: {self.processed_data_path}")
            
            return processed_data
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['data_processing'] = {
                'status': 'failed',
                'duration': stage_duration,
                'error': str(e)
            }
            logger.error(f"❌ Data processing failed: {str(e)}")
            raise
    
    def _run_feature_engineering(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering stage"""
        stage_start = time.time()
        
        try:
            logger.info("Initializing feature engineering...")
            
            # Initialize feature engineer
            feature_engineer = WalmartFeatureEngineer(processed_data)
            
            # Create all features at once
            logger.info("Creating all features...")
            numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
            categorical_cols = ['Store', 'Holiday_Flag']
            featured_data = feature_engineer.create_all_features(
                target_col='Weekly_Sales',
                date_col='Date',
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols
            )
            
            # Save featured data
            logger.info(f"Saving featured data to {self.featured_data_path}")
            featured_data.to_csv(self.featured_data_path, index=False)
            
            # Record stage results
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['feature_engineering'] = {
                'status': 'success',
                'duration': stage_duration,
                'input_rows': len(processed_data),
                'output_rows': len(featured_data),
                'input_columns': len(processed_data.columns),
                'output_columns': len(featured_data.columns),
                'features_added': len(featured_data.columns) - len(processed_data.columns),
                'output_file': str(self.featured_data_path)
            }
            
            # Data info
            self.pipeline_results['data_info']['featured'] = {
                'rows': len(featured_data),
                'columns': len(featured_data.columns),
                'columns_list': list(featured_data.columns),
                'features_added': len(featured_data.columns) - len(processed_data.columns),
                'memory_usage_mb': featured_data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"✅ Feature engineering completed in {stage_duration:.2f}s")
            logger.info(f"   • Input: {len(processed_data)} rows, {len(processed_data.columns)} columns")
            logger.info(f"   • Output: {len(featured_data)} rows, {len(featured_data.columns)} columns")
            logger.info(f"   • Features added: {len(featured_data.columns) - len(processed_data.columns)}")
            logger.info(f"   • Saved to: {self.featured_data_path}")
            
            return featured_data
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['feature_engineering'] = {
                'status': 'failed',
                'duration': stage_duration,
                'error': str(e)
            }
            logger.error(f"❌ Feature engineering failed: {str(e)}")
            raise
    
    def _run_model_training(self, featured_data: pd.DataFrame) -> dict:
        """Run model training stage using full models with cross-validation"""
        stage_start = time.time()
        
        try:
            if not MLFLOW_AVAILABLE:
                logger.info("MLflow not available - running simple model training without MLflow...")
                return self._run_simple_model_training(featured_data)
            
            logger.info("Initializing full model training with cross-validation...")
            
            # Prepare data for training (convert to format expected by train_models.py)
            training_data = self._prepare_training_data(featured_data)
            
            if training_data.empty:
                logger.warning("No training data available after preparation")
                self.pipeline_results['stages']['model_training'] = {
                    'status': 'skipped',
                    'reason': 'No training data available'
                }
                return {}
            
            logger.info(f"Prepared training data: {len(training_data)} rows")
            
            # Initialize the full model trainer
            trainer = ModelTrainer()
            
            # Train all models on the entire dataset with cross-validation
            model_results = self._train_all_models_with_cv(trainer, training_data)
            
            # Save model results
            model_results_path = self.models_dir / "training_results.json"
            with open(model_results_path, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
            
            # Record stage results
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['model_training'] = {
                'status': 'success',
                'duration': stage_duration,
                'total_data_rows': len(training_data),
                'models_trained': list(model_results.keys()),
                'successful_trainings': sum(1 for r in model_results.values() if 'error' not in r),
                'failed_trainings': sum(1 for r in model_results.values() if 'error' in r),
                'results_file': str(model_results_path)
            }
            
            self.pipeline_results['model_results'] = model_results
            
            logger.info(f"✅ Model training completed in {stage_duration:.2f}s")
            logger.info(f"   • Total data rows: {len(training_data)}")
            logger.info(f"   • Models trained: {len(model_results)}")
            logger.info(f"   • Successful: {sum(1 for r in model_results.values() if 'error' not in r)}")
            logger.info(f"   • Failed: {sum(1 for r in model_results.values() if 'error' in r)}")
            logger.info(f"   • Results saved to: {model_results_path}")
            
            return model_results
                
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['model_training'] = {
                'status': 'failed',
                'duration': stage_duration,
                'error': str(e)
            }
            logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def _run_simple_model_training(self, featured_data: pd.DataFrame) -> dict:
        """Run simple model training without MLflow dependencies"""
        stage_start = time.time()
        
        try:
            logger.info("Running simple model training without MLflow...")
            
            # Prepare data for simple training
            training_data = self._prepare_training_data(featured_data)
            
            if training_data.empty:
                logger.warning("No training data available after preparation")
                return {'status': 'skipped', 'reason': 'No training data available'}
            
            logger.info(f"Prepared training data: {len(training_data)} rows")
            
            # Simple model training without MLflow
            model_results = self._train_simple_models(training_data)
            
            # Save model results
            model_results_path = self.models_dir / "simple_training_results.json"
            with open(model_results_path, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
            
            # Record stage results
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['model_training'] = {
                'status': 'success',
                'duration': stage_duration,
                'total_data_rows': len(training_data),
                'models_trained': list(model_results.keys()),
                'successful_trainings': sum(1 for r in model_results.values() if 'error' not in r),
                'failed_trainings': sum(1 for r in model_results.values() if 'error' in r),
                'results_file': str(model_results_path),
                'note': 'Simple training without MLflow'
            }
            
            self.pipeline_results['model_results'] = model_results
            
            logger.info(f"✅ Simple model training completed in {stage_duration:.2f}s")
            logger.info(f"   • Total data rows: {len(training_data)}")
            logger.info(f"   • Models trained: {len(model_results)}")
            logger.info(f"   • Successful: {sum(1 for r in model_results.values() if 'error' not in r)}")
            logger.info(f"   • Results saved to: {model_results_path}")
            
            return model_results
                
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['model_training'] = {
                'status': 'failed',
                'duration': stage_duration,
                'error': str(e)
            }
            logger.error(f"❌ Simple model training failed: {str(e)}")
            raise
    
    def _train_simple_models(self, training_data: pd.DataFrame) -> dict:
        """Train simple models without MLflow dependencies"""
        try:
            logger.info("Training simple models...")
            
            model_results = {}
            
            # Simple Linear Regression
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_absolute_percentage_error
                from sklearn.model_selection import train_test_split
                
                # Prepare features
                feature_cols = ['temperature', 'fuel_price', 'cpi', 'unemployment_rate', 'is_holiday']
                target_col = 'weekly_sales'
                
                # Filter available columns
                available_features = [col for col in feature_cols if col in training_data.columns]
                if target_col not in training_data.columns:
                    available_features = []
                
                if available_features and len(available_features) > 0:
                    # Prepare data
                    X = training_data[available_features].fillna(0)
                    y = training_data[target_col].fillna(method='ffill')
                    
                    # Remove rows with NaN values
                    mask = ~(X.isnull().any(axis=1) | y.isnull())
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) > 10:  # Need at least some data
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Train model
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                        
                        model_results['linear_regression'] = {
                            'status': 'success',
                            'model_type': 'linear_regression',
                            'mape': mape,
                            'features_used': available_features,
                            'training_samples': len(X_train),
                            'test_samples': len(X_test),
                            'coefficients': dict(zip(available_features, model.coef_))
                        }
                        
                        logger.info(f"✅ Linear Regression trained successfully (MAPE: {mape:.2f}%)")
                    else:
                        model_results['linear_regression'] = {
                            'status': 'failed',
                            'error': 'Insufficient data for training'
                        }
                else:
                    model_results['linear_regression'] = {
                        'status': 'failed',
                        'error': 'Required features not available'
                    }
                    
            except ImportError as e:
                model_results['linear_regression'] = {
                    'status': 'failed',
                    'error': f'Scikit-learn not available: {str(e)}'
                }
            except Exception as e:
                model_results['linear_regression'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Simple Time Series Forecast (Moving Average)
            try:
                if 'weekly_sales' in training_data.columns:
                    sales_series = training_data['weekly_sales'].fillna(method='ffill')
                    
                    if len(sales_series) > 0:
                        # Simple moving average forecast
                        window_size = min(4, len(sales_series) // 2)
                        if window_size > 0:
                            moving_avg = sales_series.rolling(window=window_size).mean()
                            last_avg = moving_avg.iloc[-1]
                            
                            # Simple trend calculation
                            if len(sales_series) >= 2:
                                trend = (sales_series.iloc[-1] - sales_series.iloc[-2]) / sales_series.iloc[-2] * 100
                                forecast = last_avg * (1 + trend/100)
                            else:
                                forecast = last_avg
                            
                            model_results['moving_average'] = {
                                'status': 'success',
                                'model_type': 'moving_average',
                                'forecast': forecast,
                                'last_average': last_avg,
                                'trend_percent': trend if len(sales_series) >= 2 else 0,
                                'window_size': window_size,
                                'data_points': len(sales_series)
                            }
                            
                            logger.info(f"✅ Moving Average forecast: {forecast:.2f}")
                        else:
                            model_results['moving_average'] = {
                                'status': 'failed',
                                'error': 'Insufficient data for moving average'
                            }
                    else:
                        model_results['moving_average'] = {
                            'status': 'failed',
                            'error': 'No sales data available'
                        }
                        
            except Exception as e:
                model_results['moving_average'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Add a note about MLflow
            model_results['note'] = 'These are simple models trained without MLflow. For full MLflow integration, ensure MLflow server is running.'
            
            return model_results
            
        except Exception as e:
            logger.error(f"Error in simple model training: {str(e)}")
            raise
    
    def _prepare_training_data(self, featured_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare featured data for training with the full models"""
        try:
            logger.info("Preparing training data for full model training...")
            
            # Create a copy to avoid modifying original data
            training_data = featured_data.copy()
            
            # Ensure Date column is datetime
            if 'Date' in training_data.columns:
                training_data['Date'] = pd.to_datetime(training_data['Date'])
                training_data = training_data.sort_values('Date')
            
            # Rename columns to match what train_models.py expects
            column_mapping = {
                'Weekly_Sales': 'weekly_sales',
                'Holiday_Flag': 'is_holiday',
                'Temperature': 'temperature',
                'Fuel_Price': 'fuel_price',
                'CPI': 'cpi',
                'Unemployment': 'unemployment_rate'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in training_data.columns:
                    training_data[new_name] = training_data[old_name]
            
            # Set date as index for time series analysis
            if 'Date' in training_data.columns:
                training_data = training_data.set_index('Date')
            
            # Handle missing values
            numeric_cols = ['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment_rate']
            for col in numeric_cols:
                if col in training_data.columns:
                    # Forward fill for time series data
                    training_data[col] = training_data[col].fillna(method='ffill')
                    # Backward fill for any remaining NaNs at the beginning
                    training_data[col] = training_data[col].fillna(method='bfill')
            
            # Ensure holiday flag is boolean
            if 'is_holiday' in training_data.columns:
                training_data['is_holiday'] = training_data['is_holiday'].astype(int)
            
            logger.info(f"Training data prepared: {len(training_data)} rows, {len(training_data.columns)} columns")
            logger.info(f"Columns: {list(training_data.columns)}")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def _train_all_models_with_cv(self, trainer: ModelTrainer, training_data: pd.DataFrame) -> dict:
        """Train all models using cross-validation on the entire dataset"""
        try:
            logger.info("Training all models with cross-validation...")
            
            model_results = {}
            
            # Train each model type with cross-validation
            model_types = ['sarimax', 'lightgbm', 'xgboost', 'prophet', 'linear_regression', 'ensemble']
            
            for model_type in model_types:
                try:
                    logger.info(f"Training {model_type} model with cross-validation...")
                    
                    # Get the training function for this model type
                    if hasattr(trainer, f'_train_{model_type}'):
                        train_func = getattr(trainer, f'_train_{model_type}')
                        
                        # Train with cross-validation
                        cv_results = cross_validate_model(
                            train_func, training_data
                        )
                        
                        model_results[model_type] = {
                            'status': 'success',
                            'cv_results': cv_results,
                            'model_type': model_type
                        }
                        
                        logger.info(f"✅ {model_type} model trained successfully with CV")
                        
                    else:
                        logger.warning(f"Training function for {model_type} not found")
                        model_results[model_type] = {
                            'status': 'failed',
                            'error': f'Training function not found for {model_type}',
                            'model_type': model_type
                        }
                        
                except Exception as e:
                    logger.error(f"Error training {model_type} model: {str(e)}")
                    model_results[model_type] = {
                        'status': 'failed',
                        'error': str(e),
                        'model_type': model_type
                    }
            
            return model_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation training: {str(e)}")
            raise
    

    def _save_pipeline_summary(self):
        """Save pipeline summary to JSON file"""
        summary_path = self.output_dir / "pipeline_summary.json"
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)
            
            logger.info(f"📋 Pipeline summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline summary: {str(e)}")
    
    def get_pipeline_status(self) -> dict:
        """Get current pipeline status"""
        return self.pipeline_results
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline execution"""
        print("\n" + "=" * 60)
        print("🏪 WALMART SALES FORECASTING PIPELINE SUMMARY")
        print("=" * 60)
        
        if 'error' in self.pipeline_results:
            print(f"❌ PIPELINE FAILED: {self.pipeline_results['error']}")
            return
        
        # Overall status
        print(f"✅ Status: Completed Successfully")
        print(f"⏱️  Total Duration: {self.pipeline_results.get('total_duration', 'N/A'):.2f}s")
        print(f"🕐 Started: {self.pipeline_results.get('start_time', 'N/A')}")
        print(f"🕐 Finished: {self.pipeline_results.get('end_time', 'N/A')}")
        
        # Stage results
        print(f"\n📊 STAGE RESULTS:")
        print("-" * 40)
        
        for stage_name, stage_result in self.pipeline_results.get('stages', {}).items():
            status = stage_result.get('status', 'unknown')
            duration = stage_result.get('duration', 0)
            
            if status == 'success':
                print(f"✅ {stage_name.replace('_', ' ').title()}: {duration:.2f}s")
            elif status == 'failed':
                print(f"❌ {stage_name.replace('_', ' ').title()}: FAILED")
            elif status == 'skipped':
                print(f"⏭️  {stage_name.replace('_', ' ').title()}: SKIPPED")
            else:
                print(f"❓ {stage_name.replace('_', ' ').title()}: {status}")
        
        # Data info
        print(f"\n📈 DATA SUMMARY:")
        print("-" * 40)
        
        if 'processed' in self.pipeline_results.get('data_info', {}):
            processed = self.pipeline_results['data_info']['processed']
            print(f"📊 Processed Data: {processed.get('rows', 'N/A')} rows, {processed.get('columns', 'N/A')} columns")
            print(f"   Date Range: {processed.get('date_range', 'N/A')}")
            print(f"   Memory Usage: {processed.get('memory_usage_mb', 'N/A'):.2f} MB")
        
        if 'featured' in self.pipeline_results.get('data_info', {}):
            featured = self.pipeline_results['data_info']['featured']
            print(f"🔧 Featured Data: {featured.get('rows', 'N/A')} rows, {featured.get('columns', 'N/A')} columns")
            print(f"   Features Added: {featured.get('features_added', 'N/A')}")
            print(f"   Memory Usage: {featured.get('memory_usage_mb', 'N/A'):.2f} MB")
        
        # Model results
        if 'model_results' in self.pipeline_results:
            model_results = self.pipeline_results['model_results']
            
            # Check if model training was skipped
            if isinstance(model_results, dict) and model_results.get('status') == 'skipped':
                print(f"\n⏭️  MODEL TRAINING:")
                print("-" * 40)
                print(f"Status: SKIPPED")
                print(f"Reason: {model_results.get('reason', 'Unknown')}")
                if 'mlflow_status' in self.pipeline_results:
                    print(f"MLflow Status: {self.pipeline_results['mlflow_status']}")
            else:
                successful = sum(1 for r in model_results.values() if isinstance(r, dict) and r.get('status') == 'success')
                failed = sum(1 for r in model_results.values() if isinstance(r, dict) and r.get('status') == 'failed')
                
                print(f"\n🤖 MODEL TRAINING:")
                print("-" * 40)
                print(f"✅ Successful: {successful}")
                print(f"❌ Failed: {failed}")
                print(f"📁 Results saved to: {self.models_dir}")
                
                # Show model performance summary
                if successful > 0:
                    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
                    print("-" * 40)
                    for model_name, result in model_results.items():
                        if isinstance(result, dict) and result.get('status') == 'success':
                            if 'cv_results' in result:
                                cv = result['cv_results']
                                if 'mean_mape' in cv:
                                    print(f"🔹 {model_name.upper()}: MAPE = {cv['mean_mape']:.2f} ± {cv['std_mape']:.2f}")
                                    print(f"   Forecast = {cv['mean_forecast']:.2f} ± {cv['std_forecast']:.2f}")
                                    print(f"   CV Folds = {len(cv['folds'])}")
                            elif 'mape' in result:
                                print(f"🔹 {model_name.upper()}: MAPE = {result['mape']:.2f}%")
                                if 'forecast' in result:
                                    print(f"   Forecast = {result['forecast']:.2f}")
                            elif 'note' in result:
                                print(f"📝 Note: {result['note']}")
        
        # MLflow status
        if 'mlflow_status' in self.pipeline_results:
            print(f"\n🔧 MLFLOW STATUS:")
            print("-" * 40)
            status = self.pipeline_results['mlflow_status']
            if status == 'enabled':
                print("✅ MLflow enabled - Full model training available")
            else:
                print("⚠️  MLflow disabled - Simple model training used")
                print("   To enable full MLflow integration:")
                print("   1. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000")
                print("   2. Or set MLFLOW_TRACKING_URI environment variable")

        # Output files
        print(f"\n📁 OUTPUT FILES:")
        print("-" * 40)
        print(f"📊 Processed Data: {self.processed_data_path}")
        print(f"🔧 Featured Data: {self.featured_data_path}")
        print(f"🤖 Models & Results: {self.models_dir}")
        print(f"📋 Pipeline Summary: {self.output_dir / 'pipeline_summary.json'}")
        print(f"📝 Pipeline Log: pipeline.log")
        
        print("\n" + "=" * 60)

def main():
    """Main function to run the pipeline"""
    print("🏪 Walmart Sales Forecasting - Full Pipeline Runner")
    print("=" * 60)
    
    # Use the project root that was already set up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check if input file exists
    input_file = os.path.join(project_root, "data_manipulation", "Walmart.csv")
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("Please ensure Walmart.csv is in the data_manipulation/ directory")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = WalmartPipelineRunner(input_file)
        results = pipeline.run_full_pipeline()
        
        # Print summary
        pipeline.print_pipeline_summary()
        
        print(f"\n🎯 Pipeline completed successfully!")
        print(f"📁 Check the 'pipeline_outputs' directory for all results")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        print("Check pipeline.log for detailed error information")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
