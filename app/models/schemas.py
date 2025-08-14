"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from decimal import Decimal

# Base Models
class StoreBase(BaseModel):
    store_id: int
    store_name: str
    store_type: str
    store_size_sqft: int
    location_state: str
    location_city: str

class DepartmentBase(BaseModel):
    dept_id: int
    dept_name: str
    dept_category: str

class SalesDataBase(BaseModel):
    store_id: int
    dept_id: int
    date: date
    weekly_sales: Decimal
    is_holiday: bool = False
    temperature: Optional[Decimal] = None
    fuel_price: Optional[Decimal] = None
    cpi: Optional[Decimal] = None
    unemployment_rate: Optional[Decimal] = None

class ForecastBase(BaseModel):
    store_id: int
    dept_id: int
    forecast_date: date
    forecast_horizon: int
    predicted_sales: Decimal
    confidence_lower: Optional[Decimal] = None
    confidence_upper: Optional[Decimal] = None
    model_name: str
    model_version: str
    mape_score: Optional[Decimal] = None

class ModelPerformanceBase(BaseModel):
    model_name: str
    model_version: str
    store_id: int
    dept_id: int
    mape_score: Decimal
    wape_score: Optional[Decimal] = None
    bias_score: Optional[Decimal] = None
    validation_date: date

# Response Models
class Store(StoreBase):
    class Config:
        from_attributes = True

class Department(DepartmentBase):
    class Config:
        from_attributes = True

class SalesData(SalesDataBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class Forecast(ForecastBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class ModelPerformance(ModelPerformanceBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Request Models
class ForecastRequest(BaseModel):
    store_id: int = Field(..., description="Store ID for forecasting")
    dept_id: int = Field(..., description="Department ID for forecasting")
    forecast_horizon: int = Field(12, description="Number of weeks to forecast", ge=1, le=52)
    model_name: Optional[str] = Field(None, description="Specific model to use")
    include_confidence: bool = Field(True, description="Include confidence intervals")

class ForecastResponse(BaseModel):
    success: bool
    forecast: Forecast
    message: str
    metadata: Optional[Dict[str, Any]] = None

class SalesDataRequest(BaseModel):
    sales_data: List[SalesDataBase]

class BulkForecastRequest(BaseModel):
    requests: List[ForecastRequest]

# Analytics Models
class SalesAnalytics(BaseModel):
    total_sales: Decimal
    avg_weekly_sales: Decimal
    sales_trend: str  # "increasing", "decreasing", "stable"
    seasonal_pattern: Optional[str] = None
    holiday_impact: Optional[Decimal] = None
    top_performing_stores: List[Dict[str, Any]]
    top_performing_departments: List[Dict[str, Any]]

class ModelPerformanceSummary(BaseModel):
    model_name: str
    model_version: str
    overall_mape: Decimal
    overall_wape: Decimal
    overall_bias: Decimal
    store_performance: List[Dict[str, Any]]
    department_performance: List[Dict[str, Any]]
    last_updated: datetime

# Dashboard Models
class DashboardMetrics(BaseModel):
    total_stores: int
    total_departments: int
    total_sales_records: int
    total_forecasts: int
    avg_forecast_accuracy: Decimal
    system_health: str  # "healthy", "warning", "critical"

class TimeSeriesData(BaseModel):
    dates: List[date]
    actual_sales: List[Decimal]
    predicted_sales: List[Decimal]
    confidence_lower: Optional[List[Decimal]] = None
    confidence_upper: Optional[List[Decimal]] = None

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)






