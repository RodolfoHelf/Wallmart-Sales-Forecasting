# MLflow Troubleshooting Guide

## Problem: Pipeline Hangs on Network Connection

If you're experiencing the following error when running the pipeline:

```
File "...urllib3\connectionpool.py", line 844, in urlopen
    retries.sleep()
...
KeyboardInterrupt
```

This is caused by MLflow trying to connect to `http://localhost:5000` when no MLflow server is running.

## Solutions

### Option 1: Run Without MLflow (Recommended for Quick Testing)

The pipeline has been updated to automatically detect when MLflow is not available and will:
- Skip the full MLflow-dependent model training
- Run simple model training instead (Linear Regression, Moving Average)
- Complete successfully without hanging

**Just run the pipeline normally - it will handle MLflow issues automatically.**

### Option 2: Start MLflow Server

If you want to use the full MLflow features:

1. **Install MLflow:**
   ```bash
   pip install mlflow
   ```

2. **Start MLflow server:**
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

3. **In another terminal, run the pipeline:**
   ```bash
   python run_pipeline.py
   ```

### Option 3: Use Local File Storage

Set environment variable to use local file storage instead of server:

```bash
# Windows PowerShell
$env:MLFLOW_TRACKING_URI="file:./mlflow/local"

# Windows Command Prompt
set MLFLOW_TRACKING_URI=file:./mlflow/local

# Then run the pipeline
python run_pipeline.py
```

### Option 4: Disable MLflow Completely

Create a `.env` file in the project root:

```env
# Disable MLflow
MLFLOW_TRACKING_URI=file:./mlflow/local
```

## What Happens Without MLflow

When MLflow is not available, the pipeline will:

✅ **Data Processing** - Complete successfully  
✅ **Feature Engineering** - Complete successfully  
⚠️ **Model Training** - Use simple models instead of full MLflow models

**Simple Models Available:**
- Linear Regression (if scikit-learn is available)
- Moving Average forecasting
- Basic trend analysis

## Expected Output

```
🏪 WALMART SALES FORECASTING PIPELINE SUMMARY
============================================================
✅ Status: Completed Successfully
⏱️  Total Duration: XX.XXs

📊 STAGE RESULTS:
----------------------------------------
✅ Data Processing: X.XXs
✅ Feature Engineering: X.XXs
✅ Model Training: X.XXs

🔧 MLFLOW STATUS:
----------------------------------------
⚠️  MLflow disabled - Simple model training used
   To enable full MLflow integration:
   1. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000
   2. Or set MLFLOW_TRACKING_URI environment variable
```

## Dependencies

**Required for basic pipeline:**
- pandas, numpy
- Basic Python libraries

**Optional for simple models:**
- scikit-learn (for Linear Regression)

**Required for full MLflow models:**
- mlflow
- All ML libraries (LightGBM, XGBoost, Prophet, etc.)

## Still Having Issues?

1. Check the `pipeline.log` file for detailed error messages
2. Ensure you have the required dependencies installed
3. Try running just the data processing part first:
   ```python
   from data_manipulation import WalmartDataProcessor
   processor = WalmartDataProcessor("data_manipulation/Walmart.csv")
   data = processor.load_csv_data()
   print(f"Loaded {len(data)} rows")
   ```


