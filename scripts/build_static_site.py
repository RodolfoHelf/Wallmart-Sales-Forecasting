#!/usr/bin/env python3
"""
Static Site Builder for Walmart Sales Forecasting Dashboard
Converts the FastAPI application to static HTML files for GitHub Pages
"""

import os
import shutil
import re
from pathlib import Path

def clean_html_content(html_content):
    """Clean and optimize HTML content for static hosting"""
    # Remove FastAPI-specific paths
    html_content = html_content.replace('/static/', './static/')
    
    # Remove any remaining API references
    html_content = html_content.replace('/docs', '#')
    html_content = html_content.replace('/redoc', '#')
    
    return html_content

def build_static_site():
    """Build the static site from the FastAPI application"""
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Copy static files
    static_dir = Path("app/static")
    docs_static_dir = docs_dir / "static"
    
    if static_dir.exists():
        if docs_static_dir.exists():
            shutil.rmtree(docs_static_dir)
        shutil.copytree(static_dir, docs_static_dir)
    
    # Create the main index.html from the FastAPI app content
    main_py_path = Path("app/main.py")
    
    if main_py_path.exists():
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the HTML content from the root endpoint
        html_match = re.search(r'return """(.*?)"""', content, re.DOTALL)
        
        if html_match:
            html_content = html_match.group(1)
            
            # Clean the HTML content
            html_content = clean_html_content(html_content)
            
            # Write to docs/index.html
            with open(docs_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print("✅ Created docs/index.html")
        else:
            print("❌ Could not extract HTML content from main.py")
    else:
        print("❌ main.py not found")
    
    # Create a simple README for the docs directory
    readme_content = """# Walmart Sales Forecasting Dashboard

This is the static version of the Walmart Sales Forecasting Dashboard, built for GitHub Pages.

## Features

- Interactive dashboard with multiple tabs
- Sales forecasting insights
- Data analysis and visualization
- Feature engineering pipeline
- Model performance metrics

## Navigation

Use the sidebar navigation to explore different sections of the dashboard:
- Home: Project overview and objectives
- Problem: Current challenges and solutions
- Data: Data sources and sample information
- EDA: Exploratory data analysis insights
- Feature Engineering: Feature creation pipeline
- Modeling: ML models and approaches
- Results: Performance metrics and business impact
- Dashboard: Interactive visualizations

## Technologies

- HTML5, CSS3, JavaScript
- Font Awesome icons
- Responsive design
- Modern UI/UX principles

Visit the main repository for the full FastAPI application and ML pipeline.
"""
    
    with open(docs_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created docs/README.md")
    
    # Create .nojekyll file to disable Jekyll processing
    nojekyll_file = docs_dir / ".nojekyll"
    nojekyll_file.touch()
    print("✅ Created docs/.nojekyll")
    
    print("\n🎉 Static site built successfully!")
    print("📁 Output directory: docs/")
    print("🚀 Ready for GitHub Pages deployment")

if __name__ == "__main__":
    build_static_site()
