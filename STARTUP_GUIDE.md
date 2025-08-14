# 🚀 Walmart Sales Forecasting Dashboard - Startup Guide

## 🎯 Project Overview

This is a comprehensive ML-powered sales forecasting dashboard for Walmart's South Atlantic Division, built with modern technologies:

- **Backend**: FastAPI (Python)
- **ML Pipeline**: MLflow for experiment tracking and model versioning
- **Monitoring**: Grafana for real-time dashboards and alerts
- **Database**: PostgreSQL for data storage
- **ML Models**: SARIMAX, LightGBM, XGBoost, Prophet, and Ensemble approaches

## 🛠️ Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **Docker & Docker Compose** installed
- **Git** for version control
- **PostgreSQL** (optional, Docker will handle this)

## 📦 Installation Steps

### 1. Clone and Setup Project

```bash
# Clone the repository
git clone <your-repo-url>
cd Wallmart-Sales-Forecasting

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your configuration
# Update database credentials, API keys, etc.
```

### 3. Start Infrastructure Services

```bash
# Start PostgreSQL, Grafana, and MLflow
docker-compose up -d

# Verify services are running
docker-compose ps
```

**Services will be available at:**
- **PostgreSQL**: `localhost:5432`
- **Grafana**: `http://localhost:3000` (admin/admin)
- **MLflow**: `http://localhost:5000`

### 4. Initialize Database

```bash
# The database will be automatically initialized by Docker
# You can verify by connecting to PostgreSQL:
docker exec -it walmart_postgres psql -U walmart_user -d walmart_forecasting
```

### 5. Process Walmart Data

```bash
# Process the Walmart.csv file and load it into the database
python data/data_processor.py
```

### 6. Train ML Models

```bash
# Train all forecasting models
python models/train_models.py
```

### 7. Start the FastAPI Application

```bash
# Start the dashboard
python run.py

# Or use uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🌐 Accessing the Dashboard

### FastAPI Dashboard
- **URL**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Grafana Monitoring
- **URL**: `http://localhost:3000`
- **Username**: `admin`
- **Password**: `admin`
- **Dashboard**: Walmart Sales Forecasting Dashboard

### MLflow Tracking
- **URL**: `http://localhost:5000`
- **Experiment**: `walmart_sales_forecasting`

## 📊 Available API Endpoints

### Core Endpoints
- `GET /` - Dashboard home page
- `GET /health` - System health check
- `GET /docs` - Interactive API documentation

### Forecasting API
- `POST /api/v1/forecast` - Generate sales forecasts
- `GET /api/v1/forecasts` - Retrieve existing forecasts

### Analytics API
- `GET /api/v1/analytics/sales` - Sales analytics and trends
- `GET /api/v1/analytics/performance` - Model performance metrics

### Data Management API
- `GET /api/v1/data/sales` - Sales data retrieval
- `GET /api/v1/stores` - Store information
- `GET /api/v1/departments` - Department information

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://walmart_user:walmart_password@localhost:5432/walmart_forecasting` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server | `http://localhost:5000` |
| `GRAFANA_URL` | Grafana dashboard URL | `http://localhost:3000` |
| `API_HOST` | FastAPI host | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8000` |
| `DEBUG` | Debug mode | `True` |

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `FORECAST_HORIZON` | Number of weeks to forecast | `12` |
| `VALIDATION_WINDOW` | Weeks for model validation | `52` |

## 📈 ML Model Training

### Available Models
1. **SARIMAX** - Time series forecasting with external regressors
2. **LightGBM** - Gradient boosting with feature engineering
3. **XGBoost** - Extreme gradient boosting
4. **Prophet** - Facebook's forecasting tool
5. **Ensemble** - Combination of multiple models

### Training Process
```bash
# Train models for all store-department combinations
python models/train_models.py

# Monitor training progress in MLflow
# Open http://localhost:5000 in your browser
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=app tests/
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build production image
docker build -t walmart-forecasting .

# Run production container
docker run -d -p 8000:8000 walmart-forecasting
```

### Environment Variables for Production
```bash
# Set production environment
export ENVIRONMENT=production
export DEBUG=False
export DATABASE_URL=your_production_db_url
export MLFLOW_TRACKING_URI=your_production_mlflow_url
```

## 📚 Development Workflow

### 1. Code Quality
```bash
# Format code
black app/ models/ data/

# Lint code
flake8 app/ models/ data/

# Type checking
mypy app/ models/ data/
```

### 2. Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes
3. Add tests
4. Run tests: `pytest`
5. Commit changes: `git commit -m "Add new feature"`
6. Push and create pull request

### 3. Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart PostgreSQL
docker-compose restart postgres
```

#### 2. MLflow Not Accessible
```bash
# Check MLflow container logs
docker logs walmart_mlflow

# Restart MLflow
docker-compose restart mlflow
```

#### 3. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

#### 4. Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Logs and Debugging
```bash
# View application logs
tail -f logs/app.log

# View Docker logs
docker-compose logs -f

# Debug mode
export DEBUG=True
python run.py
```

## 📞 Support and Resources

### Documentation
- **FastAPI**: https://fastapi.tiangolo.com/
- **MLflow**: https://mlflow.org/docs/
- **Grafana**: https://grafana.com/docs/
- **PostgreSQL**: https://www.postgresql.org/docs/

### Community
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share ideas
- **Wiki**: Additional documentation and examples

## 🎉 Next Steps

1. **Explore the Dashboard**: Visit `http://localhost:8000`
2. **Train Models**: Run `python models/train_models.py`
3. **Monitor Performance**: Check Grafana at `http://localhost:3000`
4. **Track Experiments**: View MLflow at `http://localhost:5000`
5. **Customize**: Modify models, add new features, or integrate with your data sources

---

**Happy Forecasting! 🚀📊**

For additional support, please refer to the main README.md or create an issue in the project repository.









