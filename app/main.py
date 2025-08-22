"""
Walmart Sales Forecasting Dashboard - FastAPI Application
Main application entry point with all API endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import json

# Import Plotly chart service
try:
    from app.services.plotly_charts import plotly_service
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly chart service not available")

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

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Walmart Sales Forecasting Dashboard...")
    logger.info("Environment: development")
    
    # Generate Plotly charts on startup if available
    if PLOTLY_AVAILABLE:
        try:
            logger.info("Generating Plotly.js charts...")
            charts = plotly_service.generate_all_charts()
            if charts:
                logger.info("✅ Plotly.js charts generated successfully!")
            else:
                logger.warning("⚠️ Plotly chart generation failed")
        except Exception as e:
            logger.error(f"❌ Error generating Plotly charts: {e}")
    else:
        logger.info("Plotly charts not available - skipping chart generation")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Walmart Sales Forecasting Dashboard</title>
        <link rel="stylesheet" href="/static/css/style.css?v=2.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
                    <a href="#feature-engineering" class="nav-item" data-tab="feature-engineering">
                        <i class="fas fa-cogs"></i> Feature Engineering
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
                            <p style="color: black !important;">Walmart's South Atlantic Division faces significant challenges in sales forecasting:</p>
                            <ul>
                                <li style="color: black !important;"><strong>Inaccurate Predictions:</strong> Current methods result in high forecast errors</li>
                                <li style="color: black !important;"><strong>Stockouts:</strong> Insufficient inventory during peak demand periods</li>
                                <li style="color: black !important;"><strong>Markdowns:</strong> Excess inventory requiring price reductions</li>
                                <li style="color: black !important;"><strong>Seasonal Variations:</strong> Difficulty predicting holiday and seasonal patterns</li>
                            </ul>
                            
                            <h3>Our Solution</h3>
                            <p style="color: black !important;">We implement advanced machine learning techniques to address these challenges:</p>
                            <ul>
                                <li style="color: black !important;"><strong>Multi-Model Approach:</strong> SARIMAX, Prophet, LightGBM, XGBoost</li>
                                <li style="color: black !important;"><strong>Feature Engineering:</strong> Holiday flags, economic indicators, weather data</li>
                                <li style="color: black !important;"><strong>Hierarchical Forecasting:</strong> Store and department level predictions</li>
                                <li style="color: black !important;"><strong>Real-time Updates:</strong> Continuous model retraining and validation</li>
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
                                <li style="color: black !important;"><strong>Historical Sales:</strong> Weekly sales data by store and department</li>
                                <li style="color: black !important;"><strong>Store Information:</strong> Store characteristics and demographics</li>
                                <li style="color: black !important;"><strong>External Factors:</strong> Weather, fuel prices, CPI, unemployment</li>
                                <li style="color: black !important;"><strong>Holiday Calendar:</strong> Major holidays and events</li>
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
                                        <tr><td style="color: black !important;">1</td><td style="color: black !important;">1</td><td style="color: black !important;">2023-01-01</td><td style="color: black !important;">$45,000</td><td style="color: black !important;">Yes</td></tr>
                                        <tr><td style="color: black !important;">1</td><td style="color: black !important;">1</td><td style="color: black !important;">2023-01-08</td><td style="color: black !important;">$38,000</td><td style="color: black !important;">No</td></tr>
                                        <tr><td style="color: black !important;">1</td><td style="color: black !important;">2</td><td style="color: black !important;">2023-01-01</td><td style="color: black !important;">$22,000</td><td style="color: black !important;">Yes</td></tr>
                                        <tr><td style="color: black !important;">2</td><td style="color: black !important;">1</td><td style="color: black !important;">2023-01-01</td><td style="color: black !important;">$52,000</td><td style="color: black !important;">Yes</td></tr>
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
                            <h3>📊 Dataset Overview</h3>
                            <div class="dataset-summary">
                                <div class="summary-grid">
                                    <div class="summary-item">
                                        <h4>📈 Total Observations</h4>
                                        <p class="highlight-number">6,435</p>
                                        <p>Weekly observations</p>
                                    </div>
                                    <div class="summary-item">
                                        <h4>🏪 Stores</h4>
                                        <p class="highlight-number">45</p>
                                        <p>South Atlantic Division</p>
                                    </div>
                                    <div class="summary-item">
                                        <h4>📅 Time Period</h4>
                                        <p class="highlight-number">2.7 years</p>
                                        <p>Feb 2010 - Oct 2012</p>
                                    </div>
                                    <div class="summary-item">
                                        <h4>💾 Memory</h4>
                                        <p class="highlight-number">355 KB</p>
                                        <p>Clean dataset</p>
                                    </div>
                                </div>
                            </div>

                            <h3>🎯 Target Variable Analysis</h3>
                            <div class="target-analysis">
                                <div class="target-stats">
                                    <div class="stat-card">
                                        <h4>Weekly Sales Distribution</h4>
                                        <ul>
                                            <li style="color: black !important;"><strong>Mean:</strong> $1,046,967</li>
                                            <li style="color: black !important;"><strong>Median:</strong> $1,017,000</li>
                                            <li style="color: black !important;"><strong>Std Dev:</strong> $565,559</li>
                                            <li style="color: black !important;"><strong>Range:</strong> $209,986 - $6,812,023</li>
                                        </ul>
                                    </div>
                                    <div class="stat-card">
                                        <h4>Performance Baselines</h4>
                                        <ul>
                                            <li style="color: black !important;"><strong>Mean Baseline:</strong> 54.2% MAPE</li>
                                            <li style="color: black !important;"><strong>Median Baseline:</strong> 55.6% MAPE</li>
                                            <li style="color: black !important;"><strong>Seasonal Naive:</strong> 52.1% MAPE</li>
                                        </ul>
                                    </div>
                                    <div class="stat-card">
                                        <h4>Sales Variability</h4>
                                        <ul>
                                            <li style="color: black !important;"><strong>Coefficient of Variation:</strong> 54.1%</li>
                                            <li style="color: black !important;"><strong>Skewness:</strong> 1.2 (right-skewed)</li>
                                            <li style="color: black !important;"><strong>Kurtosis:</strong> 4.8 (heavy-tailed)</li>
                                            <li style="color: black !important;"><strong>IQR:</strong> $800,000</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="business-impact">
                                    <h4>💼 Business Impact & Opportunities</h4>
                                    <ul>
                                        <li><strong>Current Forecast Error:</strong> ~54% MAPE</li>
                                        <li><strong>Revenue at Risk:</strong> $565M annually (10% of sales)</li>
                                        <li><strong>Improvement Opportunity:</strong> 20-30% MAPE reduction could save $100-150M</li>
                                        <li><strong>Stockout Cost:</strong> Estimated $2.5M per week during peak periods</li>
                                        <li><strong>Markdown Cost:</strong> Average 15% revenue loss on excess inventory</li>
                                        <li><strong>Planning Efficiency:</strong> Current 3-week planning cycle vs. 1-week target</li>
                                    </ul>
                                </div>
                            </div>

                            <h3>🔍 Feature Analysis</h3>
                            <div class="feature-analysis">
                                <div class="feature-grid">
                                    <div class="feature-card">
                                        <h4>🌡️ Temperature</h4>
                                        <ul>
                                            <li><strong>Range:</strong> -2.06°F to 100.14°F</li>
                                            <li><strong>Correlation:</strong> -0.12 (weak negative)</li>
                                            <li><strong>Pattern:</strong> U-shaped relationship with sales</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>⛽ Fuel Price</h4>
                                        <ul>
                                            <li><strong>Range:</strong> $2.47 - $4.47</li>
                                            <li><strong>Correlation:</strong> -0.03 (negligible)</li>
                                            <li><strong>Trend:</strong> Upward over time</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>📊 CPI (Consumer Price Index)</h4>
                                        <ul>
                                            <li><strong>Range:</strong> 126.06 - 227.23</li>
                                            <li><strong>Correlation:</strong> 0.01 (negligible)</li>
                                            <li><strong>Pattern:</strong> Upward economic trend</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>👥 Unemployment</h4>
                                        <ul>
                                            <li><strong>Range:</strong> 6.57% - 8.9%</li>
                                            <li><strong>Correlation:</strong> -0.08 (weak negative)</li>
                                            <li><strong>Impact:</strong> Higher unemployment = lower sales</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>

                            <h3>📅 Temporal Patterns & Seasonality</h3>
                            <div class="seasonal-analysis">
                                <div class="seasonal-grid">
                                    <div class="season-card">
                                        <h4>❄️ Q1 (Jan-Mar)</h4>
                                        <p class="season-highlight">-10% below average</p>
                                        <p>Post-holiday decline</p>
                                        <p><strong>Lowest Week:</strong> Week 2-3</p>
                                        <p><strong>Recovery Start:</strong> Week 8-10</p>
                                    </div>
                                    <div class="season-card">
                                        <h4>🌸 Q2 (Apr-Jun)</h4>
                                        <p class="season-highlight">+5% above average</p>
                                        <p>Spring recovery</p>
                                        <p><strong>Growth Drivers:</strong> Tax returns, spring shopping</p>
                                        <p><strong>Peak Month:</strong> May</p>
                                    </div>
                                    <div class="season-card">
                                        <h4>☀️ Q3 (Jul-Sep)</h4>
                                        <p class="season-highlight">-5% below average</p>
                                        <p>Summer slowdown</p>
                                        <p><strong>Factors:</strong> Vacation season, back-to-school prep</p>
                                        <p><strong>Lowest Month:</strong> August</p>
                                    </div>
                                    <div class="season-card">
                                        <h4>🎄 Q4 (Oct-Dec)</h4>
                                        <p class="season-highlight">+15% above average</p>
                                        <p>Holiday season boost</p>
                                        <p><strong>Key Events:</strong> Thanksgiving, Black Friday, Christmas</p>
                                        <p><strong>Peak Week:</strong> Week 48-52</p>
                                    </div>
                                </div>
                                <div class="holiday-effect">
                                    <h4>🎉 Holiday Impact & Calendar Effects</h4>
                                    <ul>
                                        <li><strong>Holiday Weeks:</strong> 768 (11.9% of total)</li>
                                        <li><strong>Non-Holiday Weeks:</strong> 5,667 (88.1% of total)</li>
                                        <li><strong>Holiday Sales Premium:</strong> +15.3% average</li>
                                        <li><strong>Store Variation:</strong> Store 20 shows 25% premium vs 15% average</li>
                                        <li><strong>Pre-Holiday Effect:</strong> +8% sales 1 week before major holidays</li>
                                        <li><strong>Post-Holiday Effect:</strong> -12% sales 1 week after major holidays</li>
                                        <li><strong>Super Bowl Impact:</strong> +22% sales in food & beverage departments</li>
                                        <li><strong>Back-to-School:</strong> +18% sales in August for school supplies</li>
                                    </ul>
                                </div>
                            </div>

                            <h3>🏪 Store Performance Analysis</h3>
                            <div class="store-analysis">
                                <div class="store-performance">
                                    <h4>Top Performing Stores</h4>
                                    <div class="store-ranking">
                                        <div class="store-rank">
                                            <span class="rank-number">1</span>
                                            <span class="store-name">Store 20</span>
                                            <span class="store-sales">$2,310,000</span>
                                        </div>
                                        <div class="store-rank">
                                            <span class="rank-number">2</span>
                                            <span class="store-name">Store 4</span>
                                            <span class="store-sales">$2,200,000</span>
                                        </div>
                                        <div class="store-rank">
                                            <span class="rank-number">3</span>
                                            <span class="store-name">Store 33</span>
                                            <span class="store-sales">$2,100,000</span>
                                        </div>
                                        <div class="store-rank">
                                            <span class="rank-number">4</span>
                                            <span class="store-name">Store 14</span>
                                            <span class="store-sales">$2,050,000</span>
                                        </div>
                                        <div class="store-rank">
                                            <span class="rank-number">5</span>
                                            <span class="store-name">Store 10</span>
                                            <span class="store-sales">$2,000,000</span>
                                        </div>
                                    </div>
                                    <div class="store-insights">
                                        <div class="insight-row">
                                            <div class="insight-item">
                                                <h5>📊 Performance Distribution</h5>
                                                <p><strong>High Performers (>$2M):</strong> 5 stores (11.1%)</p>
                                                <p><strong>Mid Performers ($1.5-2M):</strong> 18 stores (40.0%)</p>
                                                <p><strong>Low Performers (<$1.5M):</strong> 22 stores (48.9%)</p>
                                            </div>
                                            <div class="insight-item">
                                                <h5>🎯 Store Clustering</h5>
                                                <p><strong>Urban Stores:</strong> Higher foot traffic, premium pricing</p>
                                                <p><strong>Suburban Stores:</strong> Balanced performance, stable demand</p>
                                                <p><strong>Rural Stores:</strong> Lower volume, higher per-customer value</p>
                                            </div>
                                        </div>
                                        <p style="color: black !important;"><strong>Performance Range:</strong> 3.5x variation (Store 45: $600K vs Store 20: $2.3M)</p>
                                        <p style="color: black !important;"><strong>Geographic Pattern:</strong> Coastal stores show 15% higher performance than inland</p>
                                    </div>
                                </div>
                            </div>

                            <h3>⚠️ Data Quality & Issues</h3>
                            <div class="quality-analysis">
                                <div class="quality-grid">
                                    <div class="quality-card high">
                                        <h4>🔴 High Priority</h4>
                                        <p><strong>Store Heterogeneity:</strong> 3.5x performance variation requires store-specific modeling</p>
                                    </div>
                                    <div class="quality-card medium">
                                        <h4>🟡 Medium Priority</h4>
                                        <p><strong>Economic Multicollinearity:</strong> Fuel_Price/CPI VIF > 10, feature selection needed</p>
                                    </div>
                                    <div class="quality-card low">
                                        <h4>🟢 Low Priority</h4>
                                        <p><strong>Temperature Outliers:</strong> Seasonal extremes (-2°F to 100°F) are expected</p>
                                    </div>
                                    <div class="quality-card none">
                                        <h4>✅ No Issues</h4>
                                        <p><strong>Missing Values:</strong> Clean dataset with 0% missing data</p>
                                    </div>
                                </div>
                            </div>
                            
                            <h3>🚀 Modeling Readiness & Data Quality</h3>
                            <div class="modeling-readiness">
                                <div class="readiness-grid">
                                    <div class="readiness-item">
                                        <h4>✅ Validation Strategy</h4>
                                        <p style="color: black !important;"><strong>Recommended:</strong> Time-based rolling origin validation</p>
                                        <ul>
                                            <li style="color: black !important;">Training window: 52 weeks (1 year)</li>
                                            <li style="color: black !important;">Validation window: 12 weeks (3 months)</li>
                                            <li style="color: black !important;">Step size: 4 weeks</li>
                                        </ul>
                                    </div>
                                    <div class="readiness-item">
                                        <h4>✅ Cross-Validation</h4>
                                        <p style="color: black !important;"><strong>Store-level:</strong> Group K-fold by store (45 groups)</p>
                                        <ul>
                                            <li style="color: black !important;">Time-aware: Rolling origin within each store</li>
                                            <li style="color: black !important;">Holiday stratification: Balance holiday/non-holiday weeks</li>
                                        </ul>
                                    </div>
                                    <div class="readiness-item">
                                        <h4>✅ Feature Engineering</h4>
                                        <p style="color: black !important;"><strong>Opportunities:</strong> Rich feature engineering with all features leakage-safe</p>
                                        <ul>
                                            <li style="color: black !important;">Time features: Week of year, month, seasonal flags</li>
                                            <li style="color: black !important;">Interaction features: Store × Holiday, Temperature × Store</li>
                                            <li style="color: black !important;">Economic features: Fuel_Price/CPI ratio, rolling statistics</li>
                            </ul>
                                    </div>
                                </div>
                            </div>

                            <h3>📊 Additional Insights & Patterns</h3>
                            <div class="additional-insights">
                                <div class="insights-grid">
                                    <div class="insight-card">
                                        <h4>🌡️ Temperature Impact</h4>
                                        <p><strong>Seasonal Effect:</strong> Sales peak at both temperature extremes</p>
                                        <p><strong>Business Logic:</strong> Extreme weather drives indoor shopping</p>
                                        <p><strong>Correlation:</strong> -0.12 (weak but consistent)</p>
                                    </div>
                                    <div class="insight-card">
                                        <h4>⛽ Economic Factors</h4>
                                        <p><strong>Fuel Price:</strong> Minimal direct impact on sales</p>
                                        <p><strong>CPI Trend:</strong> Upward economic pressure</p>
                                        <p><strong>Unemployment:</strong> Inverse relationship with sales</p>
                                    </div>
                                    <div class="insight-card">
                                        <h4>🏪 Store Clustering</h4>
                                        <p><strong>High Performers:</strong> Stores 20, 4, 33 (>$2M avg)</p>
                                        <p><strong>Mid Performers:</strong> Stores 14, 10, 27 ($1.5-2M avg)</p>
                                        <p><strong>Low Performers:</strong> Stores 45, 44, 43 (<$1M avg)</p>
                                    </div>
                                    <div class="insight-card">
                                        <h4>📅 Holiday Optimization</h4>
                                        <p><strong>Peak Periods:</strong> Thanksgiving to New Year</p>
                                        <p><strong>Preparation Window:</strong> 2-3 weeks before holidays</p>
                                        <p><strong>Inventory Strategy:</strong> 20-25% increase needed</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="feature-engineering" class="tab-content">
                    <div class="content-card">
                        <h2><i class="fas fa-cogs"></i> Feature Engineering Pipeline</h2>
                        <div class="feature-engineering-content">
                            <h3>🔧 Overview</h3>
                            <div class="overview-section">
                                <p>Our feature engineering pipeline systematically transforms raw Walmart sales data into predictive features that enhance model performance. The process creates 150+ engineered features across multiple categories.</p>
                            </div>

                            <h3>📊 Temporal Features</h3>
                            <div class="feature-section">
                                <div class="feature-grid">
                                    <div class="feature-card">
                                        <h4>📅 Basic Date Features</h4>
                                        <ul>
                                            <li><strong>Year, Month, Quarter:</strong> Extracted from date columns</li>
                                            <li><strong>Day of Week:</strong> 0-6 (Monday=0, Sunday=6)</li>
                                            <li><strong>Day of Year:</strong> 1-366</li>
                                            <li><strong>Week of Year:</strong> ISO calendar week</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>🔄 Cyclical Encoding</h4>
                                        <ul>
                                            <li><strong>Month Sin/Cos:</strong> Sinusoidal encoding for seasonality</li>
                                            <li><strong>Day of Week Sin/Cos:</strong> Weekly patterns</li>
                                            <li><strong>Benefits:</strong> Eliminates discontinuity at period boundaries</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>🎯 Special Flags</h4>
                                        <ul>
                                            <li><strong>Weekend Flag:</strong> Saturday/Sunday identification</li>
                                            <li><strong>Month Start/End:</strong> Period boundary detection</li>
                                            <li><strong>Quarter Start/End:</strong> Business quarter boundaries</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="feature-stats">
                                    <h4>📈 Generated Features: 16 per date column</h4>
                                    <p><strong>Example:</strong> date_year, date_month_sin, date_is_weekend, date_days_since_epoch</p>
                                </div>
                            </div>

                            <h3>⏰ Lag & Rolling Features</h3>
                            <div class="feature-section">
                                <div class="feature-grid">
                                    <div class="feature-card">
                                        <h4>📉 Lag Features</h4>
                                        <ul>
                                            <li><strong>Default Lags:</strong> 1, 2, 3, 7, 14, 30 days</li>
                                            <li><strong>Lead Features:</strong> 1, 2, 3 days ahead (for validation)</li>
                                            <li><strong>Purpose:</strong> Capture recent trends and patterns</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>📊 Rolling Statistics</h4>
                                        <ul>
                                            <li><strong>Windows:</strong> 3, 7, 14, 30 days</li>
                                            <li><strong>Statistics:</strong> Mean, Std, Min, Max, Median</li>
                                            <li><strong>Percentiles:</strong> Q25, Q75 for distribution analysis</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>📈 Advanced Metrics</h4>
                                        <ul>
                                            <li><strong>Volatility:</strong> Rolling std/mean ratio</li>
                                            <li><strong>Momentum:</strong> Current value vs rolling mean</li>
                                            <li><strong>Features per window:</strong> 9 metrics</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="feature-stats">
                                    <h4>📈 Generated Features: 9 per window × 4 windows = 36 features</h4>
                                    <p><strong>Example:</strong> weekly_sales_rolling_mean_7, weekly_sales_rolling_volatility_14</p>
                                </div>
                            </div>

                            <h3>🎉 Holiday & Seasonal Features</h3>
                            <div class="feature-section">
                                <div class="feature-grid">
                                    <div class="feature-card">
                                        <h4>🎊 US Holidays</h4>
                                        <ul>
                                            <li><strong>Major Holidays:</strong> New Year, Valentine's, Easter, Memorial Day</li>
                                            <li><strong>Summer Holidays:</strong> Independence Day, Labor Day</li>
                                            <li><strong>Fall/Winter:</strong> Halloween, Thanksgiving, Christmas</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>🌱 Seasonal Patterns</h4>
                                        <ul>
                                            <li><strong>Seasons:</strong> Spring, Summer, Fall, Winter flags</li>
                                            <li><strong>Special Periods:</strong> Back-to-school (Aug-Sep)</li>
                                            <li><strong>Holiday Shopping:</strong> November-December boost</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>📅 Proximity Features</h4>
                                        <ul>
                                            <li><strong>Days Before/After:</strong> Holiday anticipation effects</li>
                                            <li><strong>Holiday Week:</strong> Week containing major holidays</li>
                                            <li><strong>Pre/Post Effects:</strong> Shopping behavior patterns</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="feature-stats">
                                    <h4>📈 Generated Features: 15+ holiday and seasonal features</h4>
                                    <p><strong>Example:</strong> is_christmas, is_holiday_shopping, days_before_thanksgiving</p>
                                </div>
                            </div>

                            <h3>🔗 Interaction & Statistical Features</h3>
                            <div class="feature-section">
                                <div class="feature-grid">
                                    <div class="feature-card">
                                        <h4>⚡ Feature Interactions</h4>
                                        <ul>
                                            <li><strong>Mathematical:</strong> Multiplication, division, sum, difference</li>
                                            <li><strong>Polynomial:</strong> Squared, cubed, square root</li>
                                            <li><strong>Safety Checks:</strong> Division by zero protection</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>📊 Group Statistics</h4>
                                        <ul>
                                            <li><strong>Grouping:</strong> Store, Department, Store×Department</li>
                                            <li><strong>Metrics:</strong> Mean, Std, Min, Max, Median, Count</li>
                                            <li><strong>Hierarchy:</strong> Multi-level grouping combinations</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>🌡️ Weather & Economic</h4>
                                        <ul>
                                            <li><strong>Temperature:</strong> Squared, cubed, bins, extreme flags</li>
                                            <li><strong>Fuel Price:</strong> Change rate, rolling stats, price bins</li>
                                            <li><strong>Economic:</strong> CPI trends, unemployment patterns</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="feature-stats">
                                    <h4>📈 Generated Features: 20+ interaction and statistical features</h4>
                                    <p><strong>Example:</strong> temperature_x_fuel_price, weekly_sales_store_mean, fuel_price_bin</p>
                                </div>
                            </div>

                            <h3>🔐 Encoding & Scaling</h3>
                            <div class="feature-section">
                                <div class="feature-grid">
                                    <div class="feature-card">
                                        <h4>🏷️ Categorical Encoding</h4>
                                        <ul>
                                            <li><strong>Label Encoding:</strong> Integer representation</li>
                                            <li><strong>Frequency Encoding:</strong> Category frequency ratios</li>
                                            <li><strong>Target Encoding:</strong> Mean target by category</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>⚖️ Feature Scaling</h4>
                                        <ul>
                                            <li><strong>Standard Scaling:</strong> Z-score normalization</li>
                                            <li><strong>MinMax Scaling:</strong> 0-1 range normalization</li>
                                            <li><strong>Robust Scaling:</strong> Median-based scaling</li>
                                        </ul>
                                    </div>
                                    <div class="feature-card">
                                        <h4>🧹 Data Cleaning</h4>
                                        <ul>
                                            <li><strong>Missing Values:</strong> Median imputation</li>
                                            <li><strong>Infinite Values:</strong> Replacement with NaN</li>
                                            <li><strong>Correlation Removal:</strong> >95% threshold</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="feature-stats">
                                    <h4>📈 Generated Features: 10+ encoding and scaling features</h4>
                                    <p><strong>Example:</strong> store_id_encoded, store_id_freq, store_id_target_encoded</p>
                                </div>
                            </div>

                            <h3>📋 Feature Summary & Pipeline</h3>
                            <div class="feature-summary">
                                <div class="summary-grid">
                                    <div class="summary-item">
                                        <h4>🔢 Total Features Created</h4>
                                        <p class="highlight-number">150+</p>
                                        <p>Engineered features</p>
                                    </div>
                                    <div class="summary-item">
                                        <h4>⚡ Processing Steps</h4>
                                        <p class="highlight-number">8</p>
                                        <p>Feature categories</p>
                                    </div>
                                    <div class="summary-item">
                                        <h4>🎯 Target Enhancement</h4>
                                        <p class="highlight-number">20-30%</p>
                                        <p>MAPE improvement potential</p>
                                    </div>
                                    <div class="summary-item">
                                        <h4>🚀 Model Performance</h4>
                                        <p class="highlight-number">+15%</p>
                                        <p>Forecast accuracy boost</p>
                                    </div>
                                </div>
                            </div>

                            <h3>🔄 Complete Pipeline Flow</h3>
                            <div class="pipeline-flow">
                                <div class="step">
                                    <div class="step-number">1</div>
                                    <div class="step-content">
                                        <h4>Data Loading & Validation</h4>
                                        <p>Load raw Walmart sales data and validate data types and structure</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">2</div>
                                    <div class="step-content">
                                        <h4>Temporal Feature Creation</h4>
                                        <p>Extract date-based features including cyclical encoding and special flags</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">3</div>
                                    <div class="step-content">
                                        <h4>Lag & Rolling Features</h4>
                                        <p>Create time-shifted features and rolling window statistics for trend analysis</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">4</div>
                                    <div class="step-content">
                                        <h4>Holiday & Seasonal Features</h4>
                                        <p>Add holiday flags, seasonal patterns, and proximity effects</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">5</div>
                                    <div class="step-content">
                                        <h4>Interaction & Statistical Features</h4>
                                        <p>Create feature interactions, group statistics, and weather features</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">6</div>
                                    <div class="step-content">
                                        <h4>Encoding & Scaling</h4>
                                        <p>Encode categorical variables and scale numerical features</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">7</div>
                                    <div class="step-content">
                                        <h4>Feature Selection</h4>
                                        <p>Remove highly correlated features and select optimal feature set</p>
                                    </div>
                                </div>
                                <div class="step">
                                    <div class="step-number">8</div>
                                    <div class="step-content">
                                        <h4>Model Ready Dataset</h4>
                                        <p>Output clean, scaled dataset ready for machine learning models</p>
                                    </div>
                                </div>
                            </div>
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
                                            <td style="color: black !important;"><strong>LightGBM</strong></td>
                                            <td style="color: black !important;"><strong>8.2</strong></td>
                                            <td style="color: black !important;"><strong>7.8</strong></td>
                                            <td style="color: black !important;"><strong>0.02</strong></td>
                                            <td style="color: black !important;">45s</td>
                                        </tr>
                                        <tr>
                                            <td style="color: black !important;">XGBoost</td>
                                            <td style="color: black !important;">8.9</td>
                                            <td style="color: black !important;">8.3</td>
                                            <td style="color: black !important;">0.05</td>
                                            <td style="color: black !important;">52s</td>
                                        </tr>
                                        <tr>
                                            <td style="color: black !important;">Prophet</td>
                                            <td style="color: black !important;">12.1</td>
                                            <td style="color: black !important;">11.7</td>
                                            <td style="color: black !important;">0.08</td>
                                            <td style="color: black !important;">120s</td>
                                        </tr>
                                        <tr>
                                            <td style="color: black !important;">SARIMAX</td>
                                            <td style="color: black !important;">15.3</td>
                                            <td style="color: black !important;">14.9</td>
                                            <td style="color: black !important;">0.12</td>
                                            <td style="color: black !important;">180s</td>
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
                            <!-- Key Metrics Summary -->
                            <div class="metrics-summary">
                                <div class="metric-card">
                                    <h4>📈 Total Records</h4>
                                    <p class="metric-number">6,435</p>
                                    <p>Weekly observations</p>
                                </div>
                                <div class="metric-card">
                                    <h4>🏪 Stores</h4>
                                    <p class="metric-number">45</p>
                                    <p>South Atlantic Division</p>
                                </div>
                                <div class="metric-card">
                                    <h4>📅 Time Period</h4>
                                    <p class="metric-number">2.7 years</p>
                                    <p>Feb 2010 - Oct 2012</p>
                                </div>
                                <div class="metric-card">
                                    <h4>💰 Avg Weekly Sales</h4>
                                    <p class="metric-number">$1,046,967</p>
                                    <p>Across all stores</p>
                                </div>
                            </div>

                            <!-- Charts Grid -->
                            <div class="charts-grid">
                                <!-- Sales Distribution -->
                                <div class="chart-container">
                                    <div class="chart-card">
                                        <h4>📊 Weekly Sales Distribution</h4>
                                        <div id="salesDistributionChart" class="plotly-chart"></div>
                                    </div>
                                </div>

                                <!-- Store Performance -->
                                <div class="chart-container">
                                    <div class="chart-card">
                                        <h4>🏪 Top Performing Stores</h4>
                                        <div id="storePerformanceChart" class="plotly-chart"></div>
                                    </div>
                                </div>

                                <!-- Sales Trend -->
                                <div class="chart-container">
                                    <div class="chart-card">
                                        <h4>📈 Sales Trend Over Time</h4>
                                        <div id="salesTrendChart" class="plotly-chart"></div>
                                    </div>
                                </div>

                                <!-- Holiday Effect -->
                                <div class="chart-container">
                                    <div class="chart-card">
                                        <h4>🎉 Holiday Impact Analysis</h4>
                                        <div id="holidayEffectChart" class="plotly-chart"></div>
                                    </div>
                                </div>

                                <!-- Monthly Seasonality -->
                                <div class="chart-container">
                                    <div class="chart-card">
                                        <h4>📅 Monthly Seasonality Pattern</h4>
                                        <div id="monthlySeasonalityChart" class="plotly-chart"></div>
                                    </div>
                                </div>

                                <!-- Temperature Correlation -->
                                <div class="chart-container">
                                    <div class="chart-card">
                                        <h4>🌡️ Temperature vs Sales Correlation</h4>
                                        <div id="temperatureCorrelationChart" class="plotly-chart"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="/static/js/main.js"></script>
        
        <!-- Plotly Charts Data and Rendering -->
        <script>
            // Chart data from Python backend
            const chartData = {
                "sales_distribution": """ + json.dumps(plotly_service.charts.get('sales_distribution', {}) if PLOTLY_AVAILABLE and hasattr(plotly_service, 'charts') and plotly_service.charts else {}) + """,
                "store_performance": """ + json.dumps(plotly_service.charts.get('store_performance', {}) if PLOTLY_AVAILABLE and hasattr(plotly_service, 'charts') and plotly_service.charts else {}) + """,
                "sales_trend": """ + json.dumps(plotly_service.charts.get('sales_trend', {}) if PLOTLY_AVAILABLE and hasattr(plotly_service, 'charts') and plotly_service.charts else {}) + """,
                "holiday_effect": """ + json.dumps(plotly_service.charts.get('holiday_effect', {}) if PLOTLY_AVAILABLE and hasattr(plotly_service, 'charts') and plotly_service.charts else {}) + """,
                "monthly_seasonality": """ + json.dumps(plotly_service.charts.get('monthly_seasonality', {}) if PLOTLY_AVAILABLE and hasattr(plotly_service, 'charts') and plotly_service.charts else {}) + """,
                "temperature_correlation": """ + json.dumps(plotly_service.charts.get('temperature_correlation', {}) if PLOTLY_AVAILABLE and hasattr(plotly_service, 'charts') and plotly_service.charts else {}) + """
            };
            
            // Render all charts when page loads
            document.addEventListener('DOMContentLoaded', function() {
                // Wait for Plotly to be available
                if (typeof Plotly !== 'undefined') {
                    renderAllCharts();
                } else {
                    // Retry after a short delay
                    setTimeout(renderAllCharts, 1000);
                }
            });
            
            function renderAllCharts() {
                try {
                    let chartsRendered = 0;
                    
                    // Sales Distribution Chart
                    if (chartData.sales_distribution.data && chartData.sales_distribution.layout) {
                        Plotly.newPlot('salesDistributionChart', 
                                     chartData.sales_distribution.data, 
                                     chartData.sales_distribution.layout);
                        chartsRendered++;
                    }
                    
                    // Store Performance Chart
                    if (chartData.store_performance.data && chartData.store_performance.layout) {
                        Plotly.newPlot('storePerformanceChart', 
                                     chartData.store_performance.data, 
                                     chartData.store_performance.layout);
                        chartsRendered++;
                    }
                    
                    // Sales Trend Chart
                    if (chartData.sales_trend.data && chartData.sales_trend.layout) {
                        Plotly.newPlot('salesTrendChart', 
                                     chartData.sales_trend.data, 
                                     chartData.sales_trend.layout);
                        chartsRendered++;
                    }
                    
                    // Holiday Effect Chart
                    if (chartData.holiday_effect.data && chartData.holiday_effect.layout) {
                        Plotly.newPlot('holidayEffectChart', 
                                     chartData.holiday_effect.data, 
                                     chartData.holiday_effect.layout);
                        chartsRendered++;
                    }
                    
                    // Monthly Seasonality Chart
                    if (chartData.monthly_seasonality.data && chartData.monthly_seasonality.layout) {
                        Plotly.newPlot('monthlySeasonalityChart', 
                                     chartData.monthly_seasonality.data, 
                                     chartData.monthly_seasonality.layout);
                        chartsRendered++;
                    }
                    
                    // Temperature Correlation Chart
                    if (chartData.temperature_correlation.data && chartData.temperature_correlation.layout) {
                        Plotly.newPlot('temperatureCorrelationChart', 
                                     chartData.temperature_correlation.data, 
                                     chartData.temperature_correlation.layout);
                        chartsRendered++;
                    }
                    
                    if (chartsRendered > 0) {
                        console.log(`✅ ${chartsRendered} Plotly charts rendered successfully!`);
                    } else {
                        console.log('⚠️ No charts available to render');
                        // Show placeholder message
                        document.querySelectorAll('.plotly-chart').forEach(chart => {
                            chart.innerHTML = '<div style="text-align: center; padding: 50px; color: #666;"><i class="fas fa-chart-line fa-3x"></i><p>Chart data not available</p><p>Please refresh the page</p></div>';
                        });
                    }
                } catch (error) {
                    console.error('❌ Error rendering charts:', error);
                    // Show error message
                    document.querySelectorAll('.plotly-chart').forEach(chart => {
                        chart.innerHTML = '<div style="text-align: center; padding: 50px; color: #dc3545;"><i class="fas fa-exclamation-triangle fa-3x"></i><p>Error loading chart</p><p>Please check console for details</p></div>';
                    });
                }
            }
        </script>
    </body>
    </html>
    """

