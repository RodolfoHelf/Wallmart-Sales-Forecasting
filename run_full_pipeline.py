#!/usr/bin/env python3
"""
Full Pipeline Script for Walmart Sales Forecasting
Runs data processing → feature engineering → model training
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data.data_processor import WalmartDataProcessor
from data.feature_engineering import WalmartFeatureEngineer
from models.train_models import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WalmartPipelineRunner:
    """Runs the complete Walmart sales forecasting pipeline"""
    
    def __init__(self, input_csv_path: str = "data/Walmart.csv"):
        """
        Initialize the pipeline runner
        
        Args:
            input_csv_path: Path to the input Walmart.csv file
        """
        self.input_csv_path = input_csv_path
        self.output_dir = Path("pipeline_outputs")
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
            'model_results': {}
        }
    
    def run_full_pipeline(self) -> dict:
        """Run the complete pipeline from data processing to model training"""
        start_time = time.time()
        self.pipeline_results['start_time'] = datetime.now().isoformat()
        
        logger.info("🚀 Starting Walmart Sales Forecasting Full Pipeline")
        logger.info("=" * 60)
        
        try:
            # Stage 1: Data Processing
            logger.info("📊 STAGE 1: Data Processing")
            logger.info("-" * 40)
            processed_data = self._run_data_processing()
            
            # Stage 2: Feature Engineering
            logger.info("🔧 STAGE 2: Feature Engineering")
            logger.info("-" * 40)
            featured_data = self._run_feature_engineering(processed_data)
            
            # Stage 3: Model Training
            logger.info("🤖 STAGE 3: Model Training")
            logger.info("-" * 40)
            model_results = self._run_model_training(featured_data)
            
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
        """Run model training stage"""
        stage_start = time.time()
        
        try:
            logger.info("Initializing model training...")
            
            # Initialize model trainer
            trainer = ModelTrainer()
            
            # Get unique store-department combinations for training
            if 'Store' in featured_data.columns and 'Dept' in featured_data.columns:
                store_dept_combinations = featured_data[['Store', 'Dept']].drop_duplicates()
                logger.info(f"Found {len(store_dept_combinations)} store-department combinations")
                
                # Train models for first few combinations (for demonstration)
                max_combinations = min(5, len(store_dept_combinations))
                logger.info(f"Training models for first {max_combinations} combinations...")
                
                model_results = {}
                for idx, (_, row) in enumerate(store_dept_combinations.head(max_combinations).iterrows()):
                    store_id = row['Store']
                    dept_id = row['Dept']
                    
                    logger.info(f"Training models for Store {store_id}, Department {dept_id} ({idx+1}/{max_combinations})")
                    
                    try:
                        # Filter data for this store-department
                        store_dept_data = featured_data[
                            (featured_data['Store'] == store_id) & 
                            (featured_data['Dept'] == dept_id)
                        ].copy()
                        
                        if len(store_dept_data) < 20:
                            logger.warning(f"Insufficient data for Store {store_id}, Dept {dept_id}: {len(store_dept_data)} rows")
                            continue
                        
                        # Train models
                        trainer.train_all_models(store_id, dept_id)
                        
                        model_results[f"store_{store_id}_dept_{dept_id}"] = {
                            'status': 'success',
                            'data_rows': len(store_dept_data),
                            'models_trained': ['sarimax', 'lightgbm', 'xgboost', 'prophet', 'ensemble']
                        }
                        
                    except Exception as e:
                        logger.error(f"Failed to train models for Store {store_id}, Dept {dept_id}: {str(e)}")
                        model_results[f"store_{store_id}_dept_{dept_id}"] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                
                # Save model results
                model_results_path = self.models_dir / "training_results.json"
                with open(model_results_path, 'w') as f:
                    json.dump(model_results, f, indent=2, default=str)
                
                # Record stage results
                stage_duration = time.time() - stage_start
                self.pipeline_results['stages']['model_training'] = {
                    'status': 'success',
                    'duration': stage_duration,
                    'combinations_processed': len(store_dept_combinations),
                    'combinations_trained': max_combinations,
                    'successful_trainings': sum(1 for r in model_results.values() if r.get('status') == 'success'),
                    'failed_trainings': sum(1 for r in model_results.values() if r.get('status') == 'failed'),
                    'results_file': str(model_results_path)
                }
                
                self.pipeline_results['model_results'] = model_results
                
                logger.info(f"✅ Model training completed in {stage_duration:.2f}s")
                logger.info(f"   • Store-Dept combinations: {len(store_dept_combinations)}")
                logger.info(f"   • Trained: {max_combinations}")
                logger.info(f"   • Successful: {sum(1 for r in model_results.values() if r.get('status') == 'success')}")
                logger.info(f"   • Failed: {sum(1 for r in model_results.values() if r.get('status') == 'failed')}")
                logger.info(f"   • Results saved to: {model_results_path}")
                
                return model_results
                
            else:
                logger.warning("Store and Dept columns not found, skipping model training")
                self.pipeline_results['stages']['model_training'] = {
                    'status': 'skipped',
                    'reason': 'Store and Dept columns not found'
                }
                return {}
                
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.pipeline_results['stages']['model_training'] = {
                'status': 'failed',
                'duration': stage_duration,
                'error': str(e)
            }
            logger.error(f"❌ Model training failed: {str(e)}")
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
            successful = sum(1 for r in model_results.values() if r.get('status') == 'success')
            failed = sum(1 for r in model_results.values() if r.get('status') == 'failed')
            
            print(f"\n🤖 MODEL TRAINING:")
            print("-" * 40)
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"📁 Results saved to: {self.models_dir}")
        
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
    
    # Check if input file exists
    input_file = "data/Walmart.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("Please ensure Walmart.csv is in the data/ directory")
        return
    
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
