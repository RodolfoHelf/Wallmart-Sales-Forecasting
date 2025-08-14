"""
Walmart Sales Forecasting Dashboard - FastAPI Application
Main application entry point with all API endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from typing import List, Optional, Dict, Any
import logging
from datetime import date, datetime, timedelta

from .config import settings
from .database import get_db, Database
from .models.schemas import (
    SalesData, Forecast, Store, Department, 
    ModelPerformance, ForecastRequest, ForecastResponse
)
from .services.forecasting_service import ForecastingService
from .services.data_service import DataService
from .services.analytics_service import AnalyticsService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Walmart Sales Forecasting Dashboard",
    description="ML-powered sales forecasting API for Walmart South Atlantic Division",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="./app/static"), name="static")

# Initialize services
forecasting_service = ForecastingService()
data_service = DataService()
analytics_service = AnalyticsService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Walmart Sales Forecasting Dashboard...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"MLflow URI: {settings.MLFLOW_TRACKING_URI}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Walmart Sales Forecasting Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #0066cc; color: white; padding: 20px; border-radius: 8px; }
            .content { margin: 20px 0; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .endpoint h3 { color: #0066cc; margin-top: 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏪 Walmart Sales Forecasting Dashboard</h1>
            <p>ML-powered sales forecasting for South Atlantic Division</p>
        </div>
        
        <div class="content">
            <h2>📊 Available Endpoints</h2>
            
            <div class="endpoint">
                <h3>📈 Forecasting</h3>
                <p><strong>POST /api/v1/forecast</strong> - Generate sales forecasts</p>
                <p><strong>GET /api/v1/forecasts</strong> - Retrieve existing forecasts</p>
            </div>
            
            <div class="endpoint">
                <h3>📊 Analytics</h3>
                <p><strong>GET /api/v1/analytics/sales</strong> - Sales analytics and trends</p>
                <p><strong>GET /api/v1/analytics/performance</strong> - Model performance metrics</p>
            </div>
            
            <div class="endpoint">
                <h3>🗄️ Data Management</h3>
                <p><strong>GET /api/v1/data/sales</strong> - Sales data retrieval</p>
                <p><strong>POST /api/v1/data/sales</strong> - Upload new sales data</p>
            </div>
            
            <div class="endpoint">
                <h3>🔧 System</h3>
                <p><strong>GET /health</strong> - System health check</p>
                <p><strong>GET /docs</strong> - Interactive API documentation</p>
            </div>
        </div>
        
        <div class="content">
            <h2>🚀 Quick Start</h2>
            <p>Use the interactive API documentation at <a href="/docs">/docs</a> to explore all endpoints.</p>
            <p>For monitoring and alerts, access Grafana at <a href="http://localhost:3000">http://localhost:3000</a></p>
            <p>For ML model tracking, access MLflow at <a href="http://localhost:5000">http://localhost:5000</a></p>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Walmart Sales Forecasting API",
        "version": "1.0.0"
    }

# API v1 endpoints
@app.post("/api/v1/forecast", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    db: Database = Depends(get_db)
):
    """Generate sales forecast for specified store and department"""
    try:
        forecast = await forecasting_service.generate_forecast(request, db)
        return ForecastResponse(
            success=True,
            forecast=forecast,
            message="Forecast generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/forecasts", response_model=List[Forecast])
async def get_forecasts(
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    dept_id: Optional[int] = Query(None, description="Filter by department ID"),
    start_date: Optional[date] = Query(None, description="Start date for forecasts"),
    end_date: Optional[date] = Query(None, description="End date for forecasts"),
    db: Database = Depends(get_db)
):
    """Retrieve forecasts with optional filtering"""
    try:
        forecasts = await forecasting_service.get_forecasts(
            store_id, dept_id, start_date, end_date, db
        )
        return forecasts
    except Exception as e:
        logger.error(f"Error retrieving forecasts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/sales")
async def get_sales_analytics(
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    dept_id: Optional[int] = Query(None, description="Filter by department ID"),
    start_date: Optional[date] = Query(None, description="Start date for analysis"),
    end_date: Optional[date] = Query(None, description="End date for analysis"),
    db: Database = Depends(get_db)
):
    """Get sales analytics and trends"""
    try:
        analytics = await analytics_service.get_sales_analytics(
            store_id, dept_id, start_date, end_date, db
        )
        return analytics
    except Exception as e:
        logger.error(f"Error retrieving sales analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/performance")
async def get_model_performance(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    dept_id: Optional[int] = Query(None, description="Filter by department ID"),
    db: Database = Depends(get_db)
):
    """Get model performance metrics"""
    try:
        performance = await analytics_service.get_model_performance(
            model_name, store_id, dept_id, db
        )
        return performance
    except Exception as e:
        logger.error(f"Error retrieving model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/sales", response_model=List[SalesData])
async def get_sales_data(
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    dept_id: Optional[int] = Query(None, description="Filter by department ID"),
    start_date: Optional[date] = Query(None, description="Start date for data"),
    end_date: Optional[date] = Query(None, description="End date for data"),
    limit: int = Query(1000, description="Maximum number of records to return"),
    db: Database = Depends(get_db)
):
    """Retrieve sales data with optional filtering"""
    try:
        sales_data = await data_service.get_sales_data(
            store_id, dept_id, start_date, end_date, limit, db
        )
        return sales_data
    except Exception as e:
        logger.error(f"Error retrieving sales data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stores", response_model=List[Store])
async def get_stores(db: Database = Depends(get_db)):
    """Get all stores"""
    try:
        stores = await data_service.get_stores(db)
        return stores
    except Exception as e:
        logger.error(f"Error retrieving stores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/departments", response_model=List[Department])
async def get_departments(db: Database = Depends(get_db)):
    """Get all departments"""
    try:
        departments = await data_service.get_departments(db)
        return departments
    except Exception as e:
        logger.error(f"Error retrieving departments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )

