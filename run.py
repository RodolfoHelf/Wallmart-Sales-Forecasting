#!/usr/bin/env python3
"""
Simple run script for Walmart Sales Forecasting Dashboard
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("🏪 Starting Walmart Sales Forecasting Dashboard...")
    print(f"📊 API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📈 MLflow tracking at: {settings.MLFLOW_TRACKING_URI}")
    print(f"📊 Grafana dashboard at: {settings.GRAFANA_URL}")
    print("🚀 Starting FastAPI server...")
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )









