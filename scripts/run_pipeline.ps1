# Walmart Sales Forecasting - Full Pipeline Runner
# Run with: .\run_pipeline.ps1

Write-Host "🏪 Walmart Sales Forecasting - Full Pipeline Runner" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""

Write-Host "📦 Installing pipeline requirements..." -ForegroundColor Yellow
pip install -r pipeline_requirements.txt

Write-Host ""
Write-Host "🚀 Starting Full Pipeline..." -ForegroundColor Yellow
Write-Host "This will run: Data Processing → Feature Engineering → Model Training" -ForegroundColor Cyan
Write-Host ""

python run_full_pipeline.py

Write-Host ""
Write-Host "✅ Pipeline execution completed!" -ForegroundColor Green
Write-Host "📁 Check the '../../outputs/pipeline_outputs' directory for results" -ForegroundColor Cyan
Read-Host "Press Enter to continue"
