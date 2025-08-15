# Walmart Sales Forecasting - Quick Model Test
# Run with: .\run_quick_test.ps1

Write-Host "🏪 Walmart Sales Forecasting - Quick Model Test" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""

Write-Host "📦 Installing minimal requirements..." -ForegroundColor Yellow
pip install -r models/quick_test_requirements.txt

Write-Host ""
Write-Host "🚀 Running Quick Model Test..." -ForegroundColor Yellow
python models/quick_test.py

Write-Host ""
Write-Host "✅ Quick test completed!" -ForegroundColor Green
Read-Host "Press Enter to continue"
