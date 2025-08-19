# Validation Module

This module provides comprehensive validation utilities for the Walmart sales forecasting pipeline.

## Functions

### `cross_validate_model(train_func, training_data, n_splits=5, test_size_ratio=0.2)`

Performs time series cross-validation for a model.

**Parameters:**
- `train_func`: Function to train the model
- `training_data`: DataFrame with training data
- `n_splits`: Number of CV folds (default: 5)
- `test_size_ratio`: Ratio of data to use for testing (default: 0.2)

**Returns:**
- Dictionary with cross-validation results including:
  - `folds`: List of fold results
  - `mean_mape`: Mean MAPE across folds
  - `std_mape`: Standard deviation of MAPE
  - `mean_forecast`: Mean forecast values
  - `std_forecast`: Standard deviation of forecasts

### `train_test_split_time_series(data, test_size_ratio=0.2, random_state=None)`

Splits time series data into training and testing sets.

**Parameters:**
- `data`: DataFrame with time series data (should be sorted by date)
- `test_size_ratio`: Ratio of data to use for testing (default: 0.2)
- `random_state`: Random seed (not used for time series)

**Returns:**
- Tuple of (train_data, test_data)

### `rolling_window_validation(data, window_size, step_size=1)`

Performs rolling window validation for time series data.

**Parameters:**
- `data`: DataFrame with time series data
- `window_size`: Size of the rolling window
- `step_size`: Step size for moving the window (default: 1)

**Returns:**
- Dictionary with validation results

### `evaluate_model_performance(actual_values, predicted_values)`

Calculates various performance metrics for model evaluation.

**Parameters:**
- `actual_values`: Array of actual values
- `predicted_values`: Array of predicted values

**Returns:**
- Dictionary with performance metrics:
  - `mape`: Mean Absolute Percentage Error
  - `mae`: Mean Absolute Error
  - `rmse`: Root Mean Square Error
  - `r_squared`: R-squared coefficient
  - `correlation`: Correlation coefficient
  - `n_samples`: Number of samples used

## Usage Example

```python
from data_manipulation.validation import cross_validate_model

# Perform cross-validation
cv_results = cross_validate_model(
    train_func=my_model_trainer,
    training_data=my_data,
    n_splits=5,
    test_size_ratio=0.2
)

print(f"Mean MAPE: {cv_results['mean_mape']:.2f} ± {cv_results['std_mape']:.2f}")
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0

## Installation

```bash
pip install -r validation_requirements.txt
```
