@echo off
echo 🏪 Walmart Sales Forecasting - Quick Model Test
echo ================================================
echo.

echo 📦 Installing minimal requirements...
pip install -r models/quick_test_requirements.txt

echo.
echo 🚀 Running Quick Model Test...
python models/quick_test.py

echo.
echo ✅ Quick test completed!
pause
