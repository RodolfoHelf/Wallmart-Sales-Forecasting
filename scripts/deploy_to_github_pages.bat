@echo off
echo ========================================
echo  Walmart Sales Forecasting Dashboard
echo  GitHub Pages Deployment Script
echo ========================================
echo.

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)
echo ✅ Python found

echo.
echo [2/5] Installing dependencies...
pip install -r config/requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed

echo.
echo [3/5] Building static site...
python scripts/build_static_site.py
if errorlevel 1 (
    echo ERROR: Failed to build static site
    pause
    exit /b 1
)
echo ✅ Static site built

echo.
echo [4/5] Checking git status...
git status
echo.

echo [5/5] Ready to deploy!
echo.
echo Next steps:
echo 1. Review the changes above
echo 2. Commit and push to GitHub:
echo    git add .
echo    git commit -m "Deploy to GitHub Pages"
echo    git push origin main
echo.
echo 3. Enable GitHub Pages in repository settings
echo 4. Wait for automatic deployment
echo.
echo Your site will be available at:
echo https://yourusername.github.io/your-repo-name
echo.
pause
