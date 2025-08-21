# Walmart Sales Forecasting Dashboard - GitHub Pages Deployment Script
# Run this script in PowerShell to deploy your dashboard

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Walmart Sales Forecasting Dashboard" -ForegroundColor Cyan
Write-Host " GitHub Pages Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ and try again" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "[2/5] Installing dependencies..." -ForegroundColor Yellow
try {
    pip install -r config/requirements.txt
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Build static site
Write-Host ""
Write-Host "[3/5] Building static site..." -ForegroundColor Yellow
try {
    python scripts/build_static_site.py
    Write-Host "✅ Static site built" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to build static site" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check git status
Write-Host ""
Write-Host "[4/5] Checking git status..." -ForegroundColor Yellow
try {
    git status
    Write-Host "✅ Git status checked" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Warning: Git not available or not a repository" -ForegroundColor Yellow
}

# Ready to deploy
Write-Host ""
Write-Host "[5/5] Ready to deploy!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Review the changes above" -ForegroundColor White
Write-Host "2. Commit and push to GitHub:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Gray
Write-Host "   git commit -m 'Deploy to GitHub Pages'" -ForegroundColor Gray
Write-Host "   git push origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Enable GitHub Pages in repository settings" -ForegroundColor White
Write-Host "4. Wait for automatic deployment" -ForegroundColor White
Write-Host ""
Write-Host "Your site will be available at:" -ForegroundColor Cyan
Write-Host "https://yourusername.github.io/your-repo-name" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to exit"
