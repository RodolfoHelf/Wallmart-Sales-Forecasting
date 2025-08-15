@echo off
echo 🏪 Walmart Sales Forecasting - Full Pipeline Runner
echo ===================================================
echo.

echo 📦 Installing pipeline requirements...
pip install -r pipeline_requirements.txt

echo.
echo 🚀 Starting Full Pipeline...
echo This will run: Data Processing → Feature Engineering → Model Training
echo.

python run_full_pipeline.py

echo.
echo ✅ Pipeline execution completed!
echo 📁 Check the 'pipeline_outputs' directory for results
pause
