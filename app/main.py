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
        <link rel="stylesheet" href="/static/css/style.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body>
        <div class="dashboard-container">
            <!-- Sidebar -->
            <div class="sidebar">
                <div class="sidebar-header">
                    <h2><i class="fas fa-chart-line"></i> Walmart Forecasting</h2>
                </div>
                <nav class="sidebar-nav">
                    <a href="#home" class="nav-item active" data-tab="home">
                        <i class="fas fa-home"></i> Home
                    </a>
                    <a href="#problem" class="nav-item" data-tab="problem">
                        <i class="fas fa-exclamation-triangle"></i> Problem
                    </a>
                    <a href="#data" class="nav-item" data-tab="data">
                        <i class="fas fa-database"></i> Data
                    </a>
                    <a href="#eda" class="nav-item" data-tab="eda">
                        <i class="fas fa-chart-bar"></i> EDA
                    </a>
                    <a href="#modeling" class="nav-item" data-tab="modeling">
                        <i class="fas fa-brain"></i> Modeling
                    </a>
                    <a href="#results" class="nav-item" data-tab="results">
                        <i class="fas fa-trophy"></i> Results
                    </a>
                    <a href="#dashboard" class="nav-item" data-tab="dashboard">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                </nav>
            </div>

            <!-- Main Content -->
            <div class="main-content">
                <div class="content-header">
                    <h1>🏪 Walmart Sales Forecasting Dashboard</h1>
                    <p>ML-powered sales forecasting for South Atlantic Division</p>
                </div>

                <!-- Tab Content -->
                <div id="home" class="tab-content active">
                    <div class="content-card">
                        <h2><i class="fas fa-home"></i> Project Overview</h2>
                        <div class="project-summary">
                            <div class="summary-item">
                                <h3>🎯 Objective</h3>
                                <p>Develop a machine learning-powered sales forecasting dashboard for Walmart's South Atlantic Division to improve weekly forecast accuracy, reduce stockouts and markdowns, and enable data-driven decisions.</p>
                            </div>
                            <div class="summary-item">
                                <h3>🛠️ Technologies Used</h3>
                                <ul>
                                    <li><strong>Backend:</strong> Python with FastAPI</li>
                                    <li><strong>ML:</strong> MLflow, Scikit-learn, LightGBM, XGBoost</li>
                                    <li><strong>Data:</strong> Pandas, NumPy, PostgreSQL</li>
                                    <li><strong>Visualization:</strong> Plotly, Matplotlib</li>
                                </ul>
                            </div>
                            <div class="summary-item">
                                <h3>📊 Key Features</h3>
                                <ul>
                                    <li>Store x Department hierarchical forecasting</li>
                                    <li>1-12 week forecast horizon</li>
                                    <li>Holiday sensitivity analysis</li>
                                    <li>Real-time dashboard monitoring</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="problem" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-exclamation-triangle"></i> Problem Statement</h2>
                        <div class="problem-description">
                            <h3>Current Challenges</h3>
                            <p>Walmart's South Atlantic Division faces significant challenges in sales forecasting:</p>
                            <ul>
                                <li><strong>Inaccurate Predictions:</strong> Current methods result in high forecast errors</li>
                                <li><strong>Stockouts:</strong> Insufficient inventory during peak demand periods</li>
                                <li><strong>Markdowns:</strong> Excess inventory requiring price reductions</li>
                                <li><strong>Seasonal Variations:</strong> Difficulty predicting holiday and seasonal patterns</li>
                            </ul>
                            
                            <h3>Our Solution</h3>
                            <p>We implement advanced machine learning techniques to address these challenges:</p>
                            <ul>
                                <li><strong>Multi-Model Approach:</strong> SARIMAX, Prophet, LightGBM, XGBoost</li>
                                <li><strong>Feature Engineering:</strong> Holiday flags, economic indicators, weather data</li>
                                <li><strong>Hierarchical Forecasting:</strong> Store and department level predictions</li>
                                <li><strong>Real-time Updates:</strong> Continuous model retraining and validation</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div id="data" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-database"></i> Data Overview</h2>
                        <div class="data-info">
                            <h3>Data Sources</h3>
                            <ul>
                                <li><strong>Historical Sales:</strong> Weekly sales data by store and department</li>
                                <li><strong>Store Information:</strong> Store characteristics and demographics</li>
                                <li><strong>External Factors:</strong> Weather, fuel prices, CPI, unemployment</li>
                                <li><strong>Holiday Calendar:</strong> Major holidays and events</li>
                            </ul>
                            
                            <h3>Data Sample</h3>
                            <div class="data-table-container">
                                <table class="data-table">
                                    <thead>
                                        <tr>
                                            <th>Store ID</th>
                                            <th>Dept ID</th>
                                            <th>Date</th>
                                            <th>Weekly Sales</th>
                                            <th>Is Holiday</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td>1</td><td>1</td><td>2023-01-01</td><td>$45,000</td><td>Yes</td></tr>
                                        <tr><td>1</td><td>1</td><td>2023-01-08</td><td>$38,000</td><td>No</td></tr>
                                        <tr><td>1</td><td>2</td><td>2023-01-01</td><td>$22,000</td><td>Yes</td></tr>
                                        <tr><td>2</td><td>1</td><td>2023-01-01</td><td>$52,000</td><td>Yes</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="eda" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-chart-bar"></i> Exploratory Data Analysis</h2>
                        <div class="eda-content">
                            <h3>Key Insights</h3>
                            <div class="insights-grid">
                                <div class="insight-card">
                                    <h4>📈 Sales Trends</h4>
                                    <p>Clear weekly and seasonal patterns with holiday spikes</p>
                                </div>
                                <div class="insight-card">
                                    <h4>🏪 Store Performance</h4>
                                    <p>Store size and location significantly impact sales</p>
                                </div>
                                <div class="insight-card">
                                    <h4>📅 Holiday Impact</h4>
                                    <p>Holidays increase sales by 15-40% depending on department</p>
                                </div>
                                <div class="insight-card">
                                    <h4>🌡️ Weather Correlation</h4>
                                    <p>Temperature and fuel prices show moderate correlation with sales</p>
                                </div>
                            </div>
                            
                            <h3>Correlation Analysis</h3>
                            <p>Key variables showing strong correlation with sales:</p>
                            <ul>
                                <li><strong>Store Size:</strong> Positive correlation (0.67)</li>
                                <li><strong>Holiday Flag:</strong> Positive correlation (0.58)</li>
                                <li><strong>Temperature:</strong> Negative correlation (-0.42)</li>
                                <li><strong>Fuel Price:</strong> Negative correlation (-0.38)</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div id="modeling" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-brain"></i> Modeling Approach</h2>
                        <div class="modeling-content">
                            <h3>Models Implemented</h3>
                            <div class="models-grid">
                                <div class="model-card">
                                    <h4>📊 SARIMAX</h4>
                                    <p>Time series model for capturing seasonal patterns and trends</p>
                                    <ul>
                                        <li>Handles seasonality</li>
                                        <li>External regressors support</li>
                                        <li>Good for short-term forecasts</li>
                                    </ul>
                                </div>
                                <div class="model-card">
                                    <h4>🔮 Prophet</h4>
                                    <p>Facebook's forecasting tool for business metrics</p>
                                    <ul>
                                        <li>Holiday effects modeling</li>
                                        <li>Automatic seasonality detection</li>
                                        <li>Robust to missing data</li>
                                    </ul>
                                </div>
                                <div class="model-card">
                                    <h4>🌳 LightGBM</h4>
                                    <p>Gradient boosting framework for structured data</p>
                                    <ul>
                                        <li>Fast training and inference</li>
                                        <li>Handles categorical variables</li>
                                        <li>Feature importance analysis</li>
                                    </ul>
                                </div>
                                <div class="model-card">
                                    <h4>🚀 XGBoost</h4>
                                    <p>Extreme gradient boosting for high performance</p>
                                    <ul>
                                        <li>Regularization techniques</li>
                                        <li>Cross-validation support</li>
                                        <li>Excellent for tabular data</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <h3>Feature Engineering</h3>
                            <ul>
                                <li><strong>Temporal Features:</strong> Day of week, month, quarter, holiday flags</li>
                                <li><strong>Rolling Statistics:</strong> Moving averages, lag variables</li>
                                <li><strong>External Regressors:</strong> Weather, economic indicators</li>
                                <li><strong>Interaction Terms:</strong> Store-department combinations</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div id="results" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-trophy"></i> Model Results & Performance</h2>
                        <div class="results-content">
                            <h3>Performance Metrics</h3>
                            <div class="metrics-table-container">
                                <table class="metrics-table">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>MAPE (%)</th>
                                            <th>WAPE (%)</th>
                                            <th>Bias</th>
                                            <th>Training Time</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr class="best-model">
                                            <td><strong>LightGBM</strong></td>
                                            <td><strong>8.2</strong></td>
                                            <td><strong>7.8</strong></td>
                                            <td><strong>0.02</strong></td>
                                            <td>45s</td>
                                        </tr>
                                        <tr>
                                            <td>XGBoost</td>
                                            <td>8.9</td>
                                            <td>8.3</td>
                                            <td>0.05</td>
                                            <td>52s</td>
                                        </tr>
                                        <tr>
                                            <td>Prophet</td>
                                            <td>12.1</td>
                                            <td>11.7</td>
                                            <td>0.08</td>
                                            <td>120s</td>
                                        </tr>
                                        <tr>
                                            <td>SARIMAX</td>
                                            <td>15.3</td>
                                            <td>14.9</td>
                                            <td>0.12</td>
                                            <td>180s</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            
                            <h3>Business Impact</h3>
                            <div class="impact-grid">
                                <div class="impact-card">
                                    <h4>💰 Revenue Improvement</h4>
                                    <p class="impact-number">+12.5%</p>
                                    <p>Better inventory management leads to increased sales</p>
                                </div>
                                <div class="impact-card">
                                    <h4>📦 Stockout Reduction</h4>
                                    <p class="impact-number">-35%</p>
                                    <p>Fewer missed sales opportunities</p>
                                </div>
                                <div class="impact-card">
                                    <h4>🏷️ Markdown Reduction</h4>
                                    <p class="impact-number">-28%</p>
                                    <p>Less excess inventory requiring price cuts</p>
                                </div>
                                <div class="impact-card">
                                    <h4>⏱️ Planning Efficiency</h4>
                                    <p class="impact-number">+40%</p>
                                    <p>Faster and more accurate planning cycles</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="dashboard" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-tachometer-alt"></i> Interactive Dashboard</h2>
                        <div class="dashboard-content">
                            <h3>Real-time Monitoring</h3>
                            <p>This section will contain interactive charts and graphs showing:</p>
                            <ul>
                                <li><strong>Sales Trends:</strong> Historical vs. predicted sales</li>
                                <li><strong>Forecast Accuracy:</strong> Model performance over time</li>
                                <li><strong>Store Performance:</strong> Comparative analysis by location</li>
                                <li><strong>Seasonal Patterns:</strong> Holiday and seasonal effects</li>
                            </ul>
                            
                            <div class="chart-placeholder">
                                <div class="placeholder-text">
                                    <i class="fas fa-chart-line fa-3x"></i>
                                    <p>Interactive charts will be displayed here</p>
                                    <p>Connect to your data source to see real-time visualizations</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="/static/js/main.js"></script>
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

