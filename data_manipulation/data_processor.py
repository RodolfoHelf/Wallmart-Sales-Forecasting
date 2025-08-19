"""
Data processor for Walmart sales data
Handles CSV loading, cleaning, and database insertion
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db_context
from app.models.database_models import SalesData, Store, Department

logger = logging.getLogger(__name__)

class WalmartDataProcessor:
    """Processor for Walmart sales data"""
    
    def __init__(self, csv_file_path: str):
        """Initialize the data processor"""
        self.csv_file_path = csv_file_path
        self.data = None
        
    def load_csv_data(self) -> pd.DataFrame:
        """Load and parse the Walmart CSV file"""
        try:
            logger.info(f"Loading data from {self.csv_file_path}")
            
            # Read CSV file
            self.data = pd.read_csv(self.csv_file_path)
            
            # Basic data info
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            # Display data types before processing
            logger.info("Data types before processing:")
            logger.info(self.data.dtypes.to_dict())
            
            # Display first few rows
            logger.info("First 5 rows:")
            logger.info(self.data.head())
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and prepare the data"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_csv_data() first.")
            
            logger.info("Cleaning data...")
            
            # Make a copy to avoid modifying original
            clean_data = self.data.copy()
            
            # Handle missing values
            clean_data = self._handle_missing_values(clean_data)
            
            # Convert data types
            clean_data = self._convert_data_types(clean_data)
            
            # Add derived features
            clean_data = self._add_derived_features(clean_data)
            
            # Remove duplicates
            clean_data = clean_data.drop_duplicates()
            
            # Final data info
            logger.info(f"Cleaned data: {len(clean_data)} records")
            logger.info(f"Data types: {clean_data.dtypes.to_dict()}")
            
            # Verify Date column conversion
            if 'Date' in clean_data.columns:
                if clean_data['Date'].dtype == 'datetime64[ns]':
                    logger.info("✅ 'Date' column successfully converted to datetime")
                    logger.info(f"Date range: {clean_data['Date'].min()} to {clean_data['Date'].max()}")
                else:
                    logger.warning(f"⚠️  'Date' column type is {clean_data['Date'].dtype}, expected datetime64[ns]")
            
            return clean_data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Check for missing values
            missing_counts = data.isnull().sum()
            logger.info(f"Missing values per column: {missing_counts.to_dict()}")
            
            # Handle missing values based on column type
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    # For numeric columns, fill with median
                    if data[column].isnull().sum() > 0:
                        median_value = data[column].median()
                        data[column].fillna(median_value, inplace=True)
                        logger.info(f"Filled missing values in {column} with median: {median_value}")
                elif data[column].dtype == 'object':
                    # For categorical columns, fill with mode
                    if data[column].isnull().sum() > 0:
                        mode_value = data[column].mode().iloc[0] if len(data[column].mode()) > 0 else 'Unknown'
                        data[column].fillna(mode_value, inplace=True)
                        logger.info(f"Filled missing values in {column} with mode: {mode_value}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types to appropriate formats"""
        try:
            # Explicitly convert the 'Date' column to datetime
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                    logger.info("Converted 'Date' column to datetime")
                    
                    # Check for any parsing errors
                    invalid_dates = data['Date'].isna().sum()
                    if invalid_dates > 0:
                        logger.warning(f"Found {invalid_dates} invalid dates in 'Date' column")
                        
                except Exception as e:
                    logger.error(f"Error converting 'Date' column to datetime: {str(e)}")
                    raise
            
            # Convert other date columns
            date_columns = [col for col in data.columns if ('date' in col.lower() or 'time' in col.lower()) and col != 'Date']
            for col in date_columns:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
            
            # Convert numeric columns
            numeric_columns = [col for col in data.columns if 'sales' in col.lower() or 'price' in col.lower() or 'rate' in col.lower()]
            for col in numeric_columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {str(e)}")
            
            # Convert boolean columns
            boolean_columns = [col for col in data.columns if 'is_' in col.lower() or 'flag' in col.lower()]
            for col in boolean_columns:
                try:
                    data[col] = data[col].astype(bool)
                    logger.info(f"Converted {col} to boolean")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to boolean: {str(e)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error converting data types: {str(e)}")
            raise
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better analysis"""
        try:
            # Add date-based features if date column exists
            date_columns = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
            if date_columns:
                date_col = date_columns[0]
                data['year'] = data[date_col].dt.year
                data['month'] = data[date_col].dt.month
                data['quarter'] = data[date_col].dt.quarter
                data['day_of_week'] = data[date_col].dt.dayofweek
                data['is_weekend'] = data[date_col].dt.dayofweek.isin([5, 6])
                
                logger.info("Added date-based features")
            else:
                logger.warning("No datetime columns found for derived features")
            
            # Add sales-based features
            sales_columns = [col for col in data.columns if 'sales' in col.lower()]
            for col in sales_columns:
                if data[col].dtype in ['int64', 'float64']:
                    # Rolling averages
                    data[f'{col}_rolling_4'] = data[col].rolling(window=4, min_periods=1).mean()
                    data[f'{col}_rolling_12'] = data[col].rolling(window=12, min_periods=1).mean()
                    
                    # Growth rates
                    data[f'{col}_growth_rate'] = data[col].pct_change()
                    
                    logger.info(f"Added features for {col}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding derived features: {str(e)}")
            raise
    
    def verify_date_conversion(self) -> bool:
        """Verify that the Date column is properly converted to datetime"""
        try:
            if self.data is None:
                logger.error("No data loaded to verify")
                return False
            
            if 'Date' not in self.data.columns:
                logger.warning("No 'Date' column found in data")
                return False
            
            date_type = self.data['Date'].dtype
            if date_type == 'datetime64[ns]':
                logger.info("✅ Date column is properly converted to datetime")
                logger.info(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
                logger.info(f"Total unique dates: {self.data['Date'].nunique()}")
                return True
            else:
                logger.error(f"❌ Date column type is {date_type}, expected datetime64[ns]")
                logger.info("Attempting to convert Date column...")
                
                try:
                    self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
                    invalid_dates = self.data['Date'].isna().sum()
                    if invalid_dates > 0:
                        logger.warning(f"Found {invalid_dates} invalid dates after conversion")
                    
                    if self.data['Date'].dtype == 'datetime64[ns]':
                        logger.info("✅ Date column successfully converted to datetime")
                        return True
                    else:
                        logger.error("Failed to convert Date column to datetime")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error converting Date column: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error verifying date conversion: {str(e)}")
            return False
    
    def analyze_data(self) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_csv_data() first.")
            
            logger.info("Performing exploratory data analysis...")
            
            analysis = {}
            
            # Basic statistics
            analysis['basic_stats'] = {
                'total_records': len(self.data),
                'total_columns': len(self.data.columns),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'duplicate_rows': self.data.duplicated().sum()
            }
            
            # Column information
            analysis['columns'] = {
                'names': list(self.data.columns),
                'types': self.data.dtypes.to_dict(),
                'missing_values': self.data.isnull().sum().to_dict()
            }
            
            # Numeric columns analysis
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                analysis['numeric_analysis'] = {
                    'columns': list(numeric_columns),
                    'descriptive_stats': self.data[numeric_columns].describe().to_dict(),
                    'correlations': self.data[numeric_columns].corr().to_dict()
                }
            
            # Categorical columns analysis
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                analysis['categorical_analysis'] = {}
                for col in categorical_columns:
                    value_counts = self.data[col].value_counts()
                    analysis['categorical_analysis'][col] = {
                        'unique_values': len(value_counts),
                        'top_values': value_counts.head(5).to_dict()
                    }
            
            # Date columns analysis
            date_columns = [col for col in self.data.columns if self.data[col].dtype == 'datetime64[ns]']
            if date_columns:
                analysis['date_analysis'] = {}
                for col in date_columns:
                    analysis['date_analysis'][col] = {
                        'min_date': self.data[col].min().strftime('%Y-%m-%d'),
                        'max_date': self.data[col].max().strftime('%Y-%m-%d'),
                        'date_range_days': (self.data[col].max() - self.data[col].min()).days
                    }
            
            # Special analysis for Date column
            if 'Date' in self.data.columns:
                analysis['date_column_analysis'] = {
                    'column_name': 'Date',
                    'data_type': str(self.data['Date'].dtype),
                    'is_datetime': self.data['Date'].dtype == 'datetime64[ns]',
                    'total_records': len(self.data['Date']),
                    'unique_dates': self.data['Date'].nunique() if self.data['Date'].dtype == 'datetime64[ns]' else 'N/A',
                    'missing_values': self.data['Date'].isnull().sum(),
                    'sample_values': self.data['Date'].head(3).tolist() if self.data['Date'].dtype == 'datetime64[ns]' else self.data['Date'].head(3).tolist()
                }
            
            logger.info("Data analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            raise
    
    def prepare_for_database(self) -> List[Dict[str, Any]]:
        """Prepare data for database insertion"""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_csv_data() first.")
            
            logger.info("Preparing data for database insertion...")
            
            # Map CSV columns to database columns
            # This mapping should be adjusted based on actual CSV structure
            column_mapping = {
                'Store': 'store_id',
                'Date': 'date',
                'Weekly_Sales': 'weekly_sales',
                'Holiday_Flag': 'is_holiday',
                'Temperature': 'temperature',
                'Fuel_Price': 'fuel_price',
                'CPI': 'cpi',
                'Unemployment': 'unemployment_rate'
            }
            
            # Rename columns to match database schema
            data_for_db = self.data.rename(columns=column_mapping)
            
            # Select only the columns we need
            db_columns = ['store_id', 'date', 'weekly_sales', 'is_holiday', 
                         'temperature', 'fuel_price', 'cpi', 'unemployment_rate']
            
            available_columns = [col for col in db_columns if col in data_for_db.columns]
            data_for_db = data_for_db[available_columns]
            
            # Convert to list of dictionaries
            records = []
            for _, row in data_for_db.iterrows():
                record = {}
                for col in available_columns:
                    value = row[col]
                    
                    # Handle different data types
                    if pd.isna(value):
                        record[col] = None
                    elif col == 'date':
                        record[col] = value.date() if hasattr(value, 'date') else value
                    elif col == 'is_holiday':
                        record[col] = bool(value)
                    elif col in ['temperature', 'fuel_price', 'cpi', 'unemployment_rate', 'weekly_sales']:
                        record[col] = float(value) if pd.notna(value) else None
                    else:
                        record[col] = value
                
                records.append(record)
            
            logger.info(f"Prepared {len(records)} records for database insertion")
            return records
            
        except Exception as e:
            logger.error(f"Error preparing data for database: {str(e)}")
            raise
    
    def insert_to_database(self, records: List[Dict[str, Any]]) -> bool:
        """Insert prepared records into the database"""
        try:
            logger.info(f"Inserting {len(records)} records into database...")
            
            with get_db_context() as db:
                # Insert records in batches
                batch_size = 1000
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    
                    # Convert to SalesData objects
                    sales_data_objects = []
                    for record in batch:
                        sales_data = SalesData(
                            store_id=record['store_id'],

                            date=record['date'],
                            weekly_sales=Decimal(str(record['weekly_sales'])) if record['weekly_sales'] else None,
                            is_holiday=record['is_holiday'],
                            temperature=Decimal(str(record['temperature'])) if record['temperature'] else None,
                            fuel_price=Decimal(str(record['fuel_price'])) if record['fuel_price'] else None,
                            cpi=Decimal(str(record['cpi'])) if record['cpi'] else None,
                            unemployment_rate=Decimal(str(record['unemployment_rate'])) if record['unemployment_rate'] else None
                        )
                        sales_data_objects.append(sales_data)
                    
                    # Insert batch
                    db.add_all(sales_data_objects)
                    db.commit()
                    
                    logger.info(f"Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
            
            logger.info("Database insertion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting data into database: {str(e)}")
            raise

def main():
    """Main function to process Walmart data"""
    try:
        # Path to Walmart CSV file
        csv_file_path = "../Walmart.csv"
        
        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return
        
        # Initialize processor
        processor = WalmartDataProcessor(csv_file_path)
        
        # Load and process data
        processor.load_csv_data()
        
        # Verify Date column conversion
        date_ok = processor.verify_date_conversion()
        if not date_ok:
            logger.warning("Date column conversion issues detected, but continuing...")
        
        clean_data = processor.clean_data()
        
        # Analyze data
        analysis = processor.analyze_data()
        logger.info("Data analysis results:")
        logger.info(analysis)
        
        # Prepare for database
        records = processor.prepare_for_database()
        
        # Insert into database
        success = processor.insert_to_database(records)
        
        if success:
            logger.info("Data processing completed successfully")
        else:
            logger.error("Data processing failed")
            
    except Exception as e:
        logger.error(f"Error in main data processing function: {str(e)}")
        raise

if __name__ == "__main__":
    main()