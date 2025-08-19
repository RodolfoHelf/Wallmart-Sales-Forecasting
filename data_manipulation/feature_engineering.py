"""
Feature Engineering Module for Walmart Sales Forecasting
Creates additional features to enhance model performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WalmartFeatureEngineer:
    """
    Feature engineering class for Walmart sales data
    Creates temporal, lag, rolling, and interaction features
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the feature engineer
        
        Args:
            data: Input DataFrame with Walmart sales data
        """
        self.data = data.copy()
        self.original_columns = list(data.columns)
        self.feature_columns = []
        self.label_encoders = {}
        self.scalers = {}
        
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Create temporal features from date columns
        
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Creating temporal features...")
        
        # Find date columns
        date_columns = [col for col in self.data.columns 
                       if self.data[col].dtype == 'datetime64[ns]' or 'date' in col.lower()]
        
        if not date_columns:
            logger.warning("No date columns found for temporal features")
            return self.data
        
        for date_col in date_columns:
            # Basic date features
            self.data[f'{date_col}_year'] = self.data[date_col].dt.year
            self.data[f'{date_col}_month'] = self.data[date_col].dt.month
            self.data[f'{date_col}_quarter'] = self.data[date_col].dt.quarter
            self.data[f'{date_col}_day_of_week'] = self.data[date_col].dt.dayofweek
            self.data[f'{date_col}_day_of_year'] = self.data[date_col].dt.dayofyear
            self.data[f'{date_col}_week_of_year'] = self.data[date_col].dt.isocalendar().week
            
            # Cyclical encoding for periodic features
            self.data[f'{date_col}_month_sin'] = np.sin(2 * np.pi * self.data[f'{date_col}_month'] / 12)
            self.data[f'{date_col}_month_cos'] = np.cos(2 * np.pi * self.data[f'{date_col}_month'] / 12)
            self.data[f'{date_col}_day_of_week_sin'] = np.sin(2 * np.pi * self.data[f'{date_col}_day_of_week'] / 7)
            self.data[f'{date_col}_day_of_week_cos'] = np.cos(2 * np.pi * self.data[f'{date_col}_day_of_week'] / 7)
            
            # Weekend and holiday proximity
            self.data[f'{date_col}_is_weekend'] = self.data[f'{date_col}_day_of_week'].isin([5, 6]).astype(int)
            self.data[f'{date_col}_is_month_start'] = self.data[date_col].dt.is_month_start.astype(int)
            self.data[f'{date_col}_is_month_end'] = self.data[date_col].dt.is_month_end.astype(int)
            self.data[f'{date_col}_is_quarter_start'] = self.data[date_col].dt.is_quarter_start.astype(int)
            self.data[f'{date_col}_is_quarter_end'] = self.data[date_col].dt.is_quarter_end.astype(int)
            
            # Days since epoch (for trend analysis)
            self.data[f'{date_col}_days_since_epoch'] = (self.data[date_col] - pd.Timestamp('1970-01-01')).dt.days
            
            self.feature_columns.extend([
                f'{date_col}_year', f'{date_col}_month', f'{date_col}_quarter',
                f'{date_col}_day_of_week', f'{date_col}_day_of_year', f'{date_col}_week_of_year',
                f'{date_col}_month_sin', f'{date_col}_month_cos',
                f'{date_col}_day_of_week_sin', f'{date_col}_day_of_week_cos',
                f'{date_col}_is_weekend', f'{date_col}_is_month_start', f'{date_col}_is_month_end',
                f'{date_col}_is_quarter_start', f'{date_col}_is_quarter_end',
                f'{date_col}_days_since_epoch'
            ])
        
        logger.info(f"Created {len(date_columns) * 16} temporal features")
        return self.data
    
    def create_lag_features(self, target_col: str, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series analysis
        
        Args:
            target_col: Target column to create lags for
            lags: List of lag periods (default: [1, 2, 3, 7, 14, 30])
            
        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14, 30]
        
        logger.info(f"Creating lag features for {target_col} with lags: {lags}")
        
        # Sort by date if available
        date_cols = [col for col in self.data.columns if 'date' in col.lower()]
        if date_cols:
            self.data = self.data.sort_values(date_cols[0]).reset_index(drop=True)
        
        # Create lag features
        for lag in lags:
            self.data[f'{target_col}_lag_{lag}'] = self.data[target_col].shift(lag)
            self.feature_columns.append(f'{target_col}_lag_{lag}')
        
        # Create lead features (future values for validation)
        for lead in [1, 2, 3]:
            self.data[f'{target_col}_lead_{lead}'] = self.data[target_col].shift(-lead)
            self.feature_columns.append(f'{target_col}_lead_{lead}')
        
        logger.info(f"Created {len(lags) + 3} lag/lead features")
        return self.data
    
    def create_rolling_features(self, target_col: str, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            target_col: Target column for rolling features
            windows: List of window sizes (default: [3, 7, 14, 30])
            
        Returns:
            DataFrame with rolling features added
        """
        if windows is None:
            windows = [3, 7, 14, 30]
        
        logger.info(f"Creating rolling features for {target_col} with windows: {windows}")
        
        for window in windows:
            # Basic rolling statistics
            self.data[f'{target_col}_rolling_mean_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).mean()
            self.data[f'{target_col}_rolling_std_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).std()
            self.data[f'{target_col}_rolling_min_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).min()
            self.data[f'{target_col}_rolling_max_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).max()
            self.data[f'{target_col}_rolling_median_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).median()
            
            # Rolling percentiles
            self.data[f'{target_col}_rolling_q25_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).quantile(0.25)
            self.data[f'{target_col}_rolling_q75_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).quantile(0.75)
            
            # Rolling volatility
            self.data[f'{target_col}_rolling_volatility_{window}'] = self.data[target_col].rolling(window=window, min_periods=1).std() / self.data[target_col].rolling(window=window, min_periods=1).mean()
            
            # Rolling momentum
            self.data[f'{target_col}_rolling_momentum_{window}'] = self.data[target_col] / self.data[target_col].rolling(window=window, min_periods=1).mean() - 1
            
            self.feature_columns.extend([
                f'{target_col}_rolling_mean_{window}', f'{target_col}_rolling_std_{window}',
                f'{target_col}_rolling_min_{window}', f'{target_col}_rolling_max_{window}',
                f'{target_col}_rolling_median_{window}', f'{target_col}_rolling_q25_{window}',
                f'{target_col}_rolling_q75_{window}', f'{target_col}_rolling_volatility_{window}',
                f'{target_col}_rolling_momentum_{window}'
            ])
        
        logger.info(f"Created {len(windows) * 9} rolling features")
        return self.data
    
    def create_expanding_features(self, target_col: str) -> pd.DataFrame:
        """
        Create expanding window features
        
        Args:
            target_col: Target column for expanding features
            
        Returns:
            DataFrame with expanding features added
        """
        logger.info(f"Creating expanding features for {target_col}")
        
        # Expanding statistics
        self.data[f'{target_col}_expanding_mean'] = self.data[target_col].expanding().mean()
        self.data[f'{target_col}_expanding_std'] = self.data[target_col].expanding().std()
        self.data[f'{target_col}_expanding_min'] = self.data[target_col].expanding().min()
        self.data[f'{target_col}_expanding_max'] = self.data[target_col].expanding().max()
        
        # Cumulative features
        self.data[f'{target_col}_cumsum'] = self.data[target_col].cumsum()
        self.data[f'{target_col}_cummax'] = self.data[target_col].cummax()
        self.data[f'{target_col}_cummin'] = self.data[target_col].cummin()
        
        # Growth rates
        self.data[f'{target_col}_growth_rate'] = self.data[target_col].pct_change()
        self.data[f'{target_col}_cumulative_growth'] = (self.data[target_col] / self.data[target_col].iloc[0] - 1)
        
        self.feature_columns.extend([
            f'{target_col}_expanding_mean', f'{target_col}_expanding_std',
            f'{target_col}_expanding_min', f'{target_col}_expanding_max',
            f'{target_col}_cumsum', f'{target_col}_cummax', f'{target_col}_cummin',
            f'{target_col}_growth_rate', f'{target_col}_cumulative_growth'
        ])
        
        logger.info("Created 9 expanding features")
        return self.data
    
    def create_holiday_features(self, date_col: str) -> pd.DataFrame:
        """
        Create holiday and seasonal features
        
        Args:
            date_col: Date column name
            
        Returns:
            DataFrame with holiday features added
        """
        logger.info(f"Creating holiday features from {date_col}")
        
        # Major US holidays
        holidays = {
            'new_year': [(1, 1)],
            'valentines': [(2, 14)],
            'easter': [(4, 15)],  # Approximate
            'memorial_day': [(5, 27)],  # Last Monday of May
            'independence_day': [(7, 4)],
            'labor_day': [(9, 2)],  # First Monday of September
            'halloween': [(10, 31)],
            'thanksgiving': [(11, 28)],  # Fourth Thursday of November
            'christmas': [(12, 25)]
        }
        
        # Create holiday flags
        for holiday_name, dates in holidays.items():
            for month, day in dates:
                self.data[f'is_{holiday_name}'] = ((self.data[f'{date_col}_month'] == month) & 
                                                  (self.data[f'{date_col}_day_of_year'] == day)).astype(int)
                self.feature_columns.append(f'is_{holiday_name}')
        
        # Holiday proximity (days before/after major holidays)
        major_holidays = ['christmas', 'thanksgiving', 'independence_day']
        for holiday in major_holidays:
            if f'is_{holiday}' in self.data.columns:
                # Days before holiday
                self.data[f'days_before_{holiday}'] = 0
                # Days after holiday
                self.data[f'days_after_{holiday}'] = 0
                
                self.feature_columns.extend([f'days_before_{holiday}', f'days_after_{holiday}'])
        
        # Seasonal features
        self.data['is_spring'] = self.data[f'{date_col}_month'].isin([3, 4, 5]).astype(int)
        self.data['is_summer'] = self.data[f'{date_col}_month'].isin([6, 7, 8]).astype(int)
        self.data['is_fall'] = self.data[f'{date_col}_month'].isin([9, 10, 11]).astype(int)
        self.data['is_winter'] = self.data[f'{date_col}_month'].isin([12, 1, 2]).astype(int)
        
        self.feature_columns.extend(['is_spring', 'is_summer', 'is_fall', 'is_winter'])
        
        # Back to school (August-September)
        self.data['is_back_to_school'] = self.data[f'{date_col}_month'].isin([8, 9]).astype(int)
        
        # Holiday shopping season (November-December)
        self.data['is_holiday_shopping'] = self.data[f'{date_col}_month'].isin([11, 12]).astype(int)
        
        self.feature_columns.extend(['is_back_to_school', 'is_holiday_shopping'])
        
        logger.info("Created holiday and seasonal features")
        return self.data
    
    def create_interaction_features(self, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Create interaction features between numeric columns
        
        Args:
            numeric_cols: List of numeric columns for interactions
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features")
        
        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication
                self.data[f'{col1}_x_{col2}'] = self.data[col1] * self.data[col2]
                
                # Division (with safety check)
                if (self.data[col2] != 0).all():
                    self.data[f'{col1}_div_{col2}'] = self.data[col1] / self.data[col2]
                    self.feature_columns.append(f'{col1}_div_{col2}')
                
                # Sum and difference
                self.data[f'{col1}_plus_{col2}'] = self.data[col1] + self.data[col2]
                self.data[f'{col1}_minus_{col2}'] = self.data[col1] - self.data[col2]
                
                self.feature_columns.extend([
                    f'{col1}_x_{col2}', f'{col1}_plus_{col2}', f'{col1}_minus_{col2}'
                ])
        
        # Polynomial features for important columns
        important_cols = [col for col in numeric_cols if 'sales' in col.lower() or 'price' in col.lower()]
        for col in important_cols[:3]:  # Limit to top 3 to avoid explosion
            self.data[f'{col}_squared'] = self.data[col] ** 2
            self.data[f'{col}_cubed'] = self.data[col] ** 3
            self.data[f'{col}_sqrt'] = np.sqrt(np.abs(self.data[col]))
            
            self.feature_columns.extend([f'{col}_squared', f'{col}_cubed', f'{col}_sqrt'])
        
        logger.info("Created interaction features")
        return self.data
    
    def create_statistical_features(self, target_col: str, group_cols: List[str]) -> pd.DataFrame:
        """
        Create statistical features grouped by categories
        
        Args:
            target_col: Target column for statistics
            group_cols: Columns to group by
            
        Returns:
            DataFrame with statistical features added
        """
        logger.info(f"Creating statistical features grouped by {group_cols}")
        
        # Group by combinations
        for i in range(1, min(4, len(group_cols) + 1)):
            for group_combo in [group_cols[j:j+i] for j in range(len(group_cols) - i + 1)]:
                group_name = '_'.join(group_combo)
                
                # Calculate group statistics
                group_stats = self.data.groupby(group_combo)[target_col].agg([
                    'mean', 'std', 'min', 'max', 'median', 'count'
                ]).reset_index()
                
                # Rename columns
                group_stats.columns = group_combo + [f'{target_col}_{group_name}_mean', 
                                                   f'{target_col}_{group_name}_std',
                                                   f'{target_col}_{group_name}_min',
                                                   f'{target_col}_{group_name}_max',
                                                   f'{target_col}_{group_name}_median',
                                                   f'{target_col}_{group_name}_count']
                
                # Merge back to original data
                self.data = self.data.merge(group_stats, on=group_combo, how='left')
                
                # Add to feature columns
                self.feature_columns.extend([
                    f'{target_col}_{group_name}_mean', f'{target_col}_{group_name}_std',
                    f'{target_col}_{group_name}_min', f'{target_col}_{group_name}_max',
                    f'{target_col}_{group_name}_median', f'{target_col}_{group_name}_count'
                ])
        
        logger.info("Created statistical features")
        return self.data
    
    def create_encoding_features(self, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Create encoding features for categorical variables
        
        Args:
            categorical_cols: List of categorical columns
            
        Returns:
            DataFrame with encoding features added
        """
        logger.info("Creating encoding features")
        
        for col in categorical_cols:
            if col in self.data.columns:
                # Label encoding
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                
                # Frequency encoding
                freq_encoding = self.data[col].value_counts(normalize=True)
                self.data[f'{col}_freq'] = self.data[col].map(freq_encoding)
                
                # Target encoding (if target column exists)
                target_cols = [col for col in self.data.columns if 'sales' in col.lower()]
                if target_cols:
                    target_col = target_cols[0]
                    target_encoding = self.data.groupby(col)[target_col].mean()
                    self.data[f'{col}_target_encoded'] = self.data[col].map(target_encoding)
                    self.feature_columns.append(f'{col}_target_encoded')
                
                self.feature_columns.extend([f'{col}_encoded', f'{col}_freq'])
        
        logger.info("Created encoding features")
        return self.data
    
    def create_weather_features(self, temp_col: str = None, fuel_col: str = None) -> pd.DataFrame:
        """
        Create weather and economic related features
        
        Args:
            temp_col: Temperature column name
            fuel_col: Fuel price column name
            
        Returns:
            DataFrame with weather features added
        """
        logger.info("Creating weather and economic features")
        
        if temp_col and temp_col in self.data.columns:
            # Temperature features
            self.data[f'{temp_col}_squared'] = self.data[temp_col] ** 2
            self.data[f'{temp_col}_cubed'] = self.data[temp_col] ** 3
            
            # Temperature bins
            self.data[f'{temp_col}_bin'] = pd.cut(self.data[temp_col], 
                                                 bins=[-np.inf, 32, 50, 68, 86, np.inf], 
                                                 labels=['very_cold', 'cold', 'cool', 'warm', 'hot'])
            
            # Temperature extremes
            self.data[f'{temp_col}_is_extreme'] = ((self.data[temp_col] < 32) | 
                                                  (self.data[temp_col] > 90)).astype(int)
            
            self.feature_columns.extend([
                f'{temp_col}_squared', f'{temp_col}_cubed', f'{temp_col}_bin', f'{temp_col}_is_extreme'
            ])
        
        if fuel_col and fuel_col in self.data.columns:
            # Fuel price features
            self.data[f'{fuel_col}_change'] = self.data[fuel_col].pct_change()
            self.data[f'{fuel_col}_rolling_mean_7'] = self.data[fuel_col].rolling(7, min_periods=1).mean()
            self.data[f'{fuel_col}_rolling_std_7'] = self.data[fuel_col].rolling(7, min_periods=1).std()
            
            # Fuel price bins
            self.data[f'{fuel_col}_bin'] = pd.qcut(self.data[fuel_col], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
            
            self.feature_columns.extend([
                f'{fuel_col}_change', f'{fuel_col}_rolling_mean_7', f'{fuel_col}_rolling_std_7', f'{fuel_col}_bin'
            ])
        
        logger.info("Created weather and economic features")
        return self.data
    
    def scale_features(self, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} method")
        
        # Get numerical feature columns
        numerical_features = [col for col in self.feature_columns 
                            if self.data[col].dtype in ['int64', 'float64']]
        
        if not numerical_features:
            logger.warning("No numerical features to scale")
            return self.data
        
        # Remove infinite and NaN values
        for col in numerical_features:
            self.data[col] = self.data[col].replace([np.inf, -np.inf], np.nan)
            self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Apply scaling
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
            scaler = StandardScaler()
        
        # Scale features
        scaled_features = scaler.fit_transform(self.data[numerical_features])
        self.data[numerical_features] = scaled_features
        self.scalers[method] = scaler
        
        logger.info(f"Scaled {len(numerical_features)} features using {method} method")
        return self.data
    
    def remove_correlated_features(self, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            threshold: Correlation threshold for removal
            
        Returns:
            DataFrame with correlated features removed
        """
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Get numerical feature columns
        numerical_features = [col for col in self.feature_columns 
                            if self.data[col].dtype in ['int64', 'float64']]
        
        if len(numerical_features) < 2:
            logger.warning("Not enough numerical features for correlation analysis")
            return self.data
        
        # Calculate correlation matrix
        corr_matrix = self.data[numerical_features].corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Remove correlated features
        if to_drop:
            self.data = self.data.drop(columns=to_drop)
            self.feature_columns = [col for col in self.feature_columns if col not in to_drop]
            logger.info(f"Removed {len(to_drop)} correlated features: {to_drop}")
        else:
            logger.info("No highly correlated features found")
        
        return self.data
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of created features
        
        Returns:
            Dictionary with feature information
        """
        summary = {
            'total_features': len(self.feature_columns),
            'original_columns': len(self.original_columns),
            'feature_columns': self.feature_columns,
            'feature_types': {},
            'missing_values': {},
            'correlation_with_target': {}
        }
        
        # Analyze feature types
        for col in self.feature_columns:
            if col in self.data.columns:
                summary['feature_types'][col] = str(self.data[col].dtype)
                summary['missing_values'][col] = self.data[col].isnull().sum()
        
        # Find target column for correlation analysis
        target_cols = [col for col in self.data.columns if 'sales' in col.lower()]
        if target_cols:
            target_col = target_cols[0]
            for col in self.feature_columns:
                if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']:
                    correlation = self.data[col].corr(self.data[target_col])
                    summary['correlation_with_target'][col] = correlation
        
        return summary
    
    def create_all_features(self, target_col: str, date_col: str, 
                           categorical_cols: List[str] = None,
                           numeric_cols: List[str] = None) -> pd.DataFrame:
        """
        Create all feature types in one go
        
        Args:
            target_col: Target column for lag and rolling features
            date_col: Date column for temporal features
            categorical_cols: List of categorical columns
            numeric_cols: List of numeric columns
            
        Returns:
            DataFrame with all features created
        """
        logger.info("Creating all features...")
        
        # Create all feature types
        self.create_temporal_features()
        self.create_lag_features(target_col)
        self.create_rolling_features(target_col)
        self.create_expanding_features(target_col)
        self.create_holiday_features(date_col)
        
        if categorical_cols:
            self.create_encoding_features(categorical_cols)
        
        if categorical_cols:
            self.create_statistical_features(target_col, categorical_cols)
        
        if numeric_cols:
            self.create_interaction_features(numeric_cols)
            # Create weather features if temperature and fuel_price columns exist
            temp_cols = [col for col in numeric_cols if 'temperature' in col.lower()]
            fuel_cols = [col for col in numeric_cols if 'fuel' in col.lower()]
            if temp_cols and fuel_cols:
                self.create_weather_features(temp_cols[0], fuel_cols[0])
        
        # Remove correlated features
        self.remove_correlated_features()
        
        # Scale features
        self.scale_features()
        
        logger.info(f"Feature engineering completed. Created {len(self.feature_columns)} features.")
        return self.data

def main():
    """Example usage of the feature engineer"""
    # Load your data
    # data = pd.read_csv('data/Walmart.csv')
    
    # Initialize feature engineer
    # engineer = WalmartFeatureEngineer(data)
    
    # Create all features
    # enhanced_data = engineer.create_all_features(
    #     target_col='weekly_sales',
    #     date_col='date',
    #     categorical_cols=['store_id', 'dept_id'],
    #     numeric_cols=['temperature', 'fuel_price', 'cpi', 'unemployment_rate']
    # )
    
    # Get feature summary
    # summary = engineer.get_feature_summary()
    # print("Feature Summary:", summary)
    
    print("Feature Engineering Module Ready!")
    print("Use WalmartFeatureEngineer class to create enhanced features for your models.")

if __name__ == "__main__":
    main()