#!/usr/bin/env python3
"""
Validation utilities for Walmart sales forecasting pipeline
Includes cross-validation and train-test validation functions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


def cross_validate_model(train_func: Callable, training_data: pd.DataFrame, 
                        n_splits: int = 5, test_size_ratio: float = 0.2) -> Dict[str, Any]:
    """
    Perform time series cross-validation for a model
    
    Args:
        train_func: Function to train the model
        training_data: DataFrame with training data
        n_splits: Number of CV folds
        test_size_ratio: Ratio of data to use for testing (0.0 to 1.0)
    
    Returns:
        Dictionary with cross-validation results
    """
    try:
        logger.info("Performing time series cross-validation...")
        
        # Calculate test size
        test_size = int(len(training_data) * test_size_ratio)
        
        cv_results = {
            'folds': [],
            'mean_mape': 0.0,
            'std_mape': 0.0,
            'mean_forecast': 0.0,
            'std_forecast': 0.0,
            'n_splits': n_splits,
            'test_size_ratio': test_size_ratio
        }
        
        # Perform time series cross-validation
        for fold in range(n_splits):
            try:
                # Calculate split indices for time series
                split_point = len(training_data) - (n_splits - fold) * test_size
                
                if split_point <= test_size:
                    continue  # Skip if not enough data
                
                # Split data
                train_data = training_data.iloc[:split_point]
                test_data = training_data.iloc[split_point:split_point + test_size]
                
                if len(train_data) < 50 or len(test_data) < 10:
                    logger.warning(f"Fold {fold}: Insufficient data for training/testing")
                    continue
                
                logger.info(f"Fold {fold + 1}: Train={len(train_data)}, Test={len(test_data)}")
                
                # Train model on this fold
                fold_result = train_func(train_data)
                
                # Evaluate on test set
                if 'error' not in fold_result:
                    # Calculate test MAPE
                    test_sales = test_data['weekly_sales'].values
                    if 'forecast_value' in fold_result:
                        forecast = fold_result['forecast_value']
                        # Simple evaluation: compare forecast with test mean
                        test_mape = np.mean(np.abs((test_sales - forecast) / test_sales)) * 100
                    else:
                        test_mape = fold_result.get('mape_score', 0)
                    
                    fold_summary = {
                        'fold': fold + 1,
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'mape': test_mape,
                        'forecast_value': fold_result.get('forecast_value', 0),
                        'model_metrics': fold_result
                    }
                    
                    cv_results['folds'].append(fold_summary)
                    
            except Exception as e:
                logger.warning(f"Fold {fold + 1} failed: {str(e)}")
                continue
        
        # Calculate summary statistics
        if cv_results['folds']:
            mapes = [fold['mape'] for fold in cv_results['folds']]
            forecasts = [fold['forecast_value'] for fold in cv_results['folds']]
            
            cv_results['mean_mape'] = float(np.mean(mapes))
            cv_results['std_mape'] = float(np.std(mapes))
            cv_results['mean_forecast'] = float(np.mean(forecasts))
            cv_results['std_forecast'] = float(np.std(forecasts))
            
            logger.info(f"Cross-validation completed: {len(cv_results['folds'])} folds")
            logger.info(f"Mean MAPE: {cv_results['mean_mape']:.2f} ± {cv_results['std_mape']:.2f}")
            logger.info(f"Mean Forecast: {cv_results['mean_forecast']:.2f} ± {cv_results['std_forecast']:.2f}")
        
        return cv_results
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return {'error': str(e)}


def train_test_split_time_series(data: pd.DataFrame, test_size_ratio: float = 0.2, 
                                random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training and testing sets
    
    Args:
        data: DataFrame with time series data (should be sorted by date)
        test_size_ratio: Ratio of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility (not used for time series)
    
    Returns:
        Tuple of (train_data, test_data)
    """
    try:
        # Ensure data is sorted by index (date)
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        
        # Calculate split point
        split_point = int(len(data) * (1 - test_size_ratio))
        
        # Split data
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        logger.info(f"Time series split: Train={len(train_data)}, Test={len(test_data)}")
        logger.info(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
        logger.info(f"Test date range: {test_data.index.min()} to {test_data.index.max()}")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error in train-test split: {str(e)}")
        raise


def rolling_window_validation(data: pd.DataFrame, window_size: int, 
                            step_size: int = 1) -> Dict[str, Any]:
    """
    Perform rolling window validation for time series data
    
    Args:
        data: DataFrame with time series data
        window_size: Size of the rolling window
        step_size: Step size for moving the window
    
    Returns:
        Dictionary with validation results
    """
    try:
        logger.info(f"Performing rolling window validation with window_size={window_size}, step_size={step_size}")
        
        validation_results = {
            'windows': [],
            'mean_mape': 0.0,
            'std_mape': 0.0,
            'window_size': window_size,
            'step_size': step_size
        }
        
        # Perform rolling window validation
        for start_idx in range(0, len(data) - window_size, step_size):
            try:
                end_idx = start_idx + window_size
                
                # Split data into window and future
                window_data = data.iloc[start_idx:end_idx]
                future_data = data.iloc[end_idx:end_idx + step_size]
                
                if len(future_data) == 0:
                    continue
                
                # Calculate simple metrics for the window
                window_sales = window_data['weekly_sales'].values
                future_sales = future_data['weekly_sales'].values
                
                # Simple forecast: use last value from window
                forecast = window_sales[-1]
                
                # Calculate MAPE for future data
                mape = np.mean(np.abs((future_sales - forecast) / future_sales)) * 100
                
                window_result = {
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'window_size': len(window_data),
                    'future_size': len(future_data),
                    'mape': mape,
                    'forecast_value': forecast,
                    'actual_values': future_sales.tolist()
                }
                
                validation_results['windows'].append(window_result)
                
            except Exception as e:
                logger.warning(f"Window {start_idx} failed: {str(e)}")
                continue
        
        # Calculate summary statistics
        if validation_results['windows']:
            mapes = [window['mape'] for window in validation_results['windows']]
            validation_results['mean_mape'] = float(np.mean(mapes))
            validation_results['std_mape'] = float(np.std(mapes))
            
            logger.info(f"Rolling window validation completed: {len(validation_results['windows'])} windows")
            logger.info(f"Mean MAPE: {validation_results['mean_mape']:.2f} ± {validation_results['std_mape']:.2f}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in rolling window validation: {str(e)}")
        return {'error': str(e)}


def evaluate_model_performance(actual_values: np.ndarray, predicted_values: np.ndarray) -> Dict[str, float]:
    """
    Calculate various performance metrics for model evaluation
    
    Args:
        actual_values: Array of actual values
        predicted_values: Array of predicted values
    
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Remove any NaN values
        mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
        actual = actual_values[mask]
        predicted = predicted_values[mask]
        
        if len(actual) == 0:
            return {'error': 'No valid data for evaluation'}
        
        # Calculate metrics
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Calculate R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate correlation
        correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
        
        return {
            'mape': float(mape),
            'mae': float(mae),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'n_samples': len(actual)
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {'error': str(e)}
