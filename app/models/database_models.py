"""
SQLAlchemy database models for Walmart Sales Forecasting
"""

from sqlalchemy import Column, Integer, String, Date, Boolean, Numeric, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from ..database import Base

class Store(Base):
    """Store information"""
    __tablename__ = "stores"
    
    store_id = Column(Integer, primary_key=True, index=True)
    store_name = Column(String(100), nullable=False)
    store_type = Column(String(50), nullable=False)
    store_size_sqft = Column(Integer, nullable=False)
    location_state = Column(String(50), nullable=False)
    location_city = Column(String(100), nullable=False)
    
    # Relationships
    sales_data = relationship("SalesData", back_populates="store")
    forecasts = relationship("Forecast", back_populates="store")
    model_performance = relationship("ModelPerformance", back_populates="store")
    
    def __repr__(self):
        return f"<Store(store_id={self.store_id}, name='{self.store_name}')>"

class Department(Base):
    """Department information"""
    __tablename__ = "departments"
    
    dept_id = Column(Integer, primary_key=True, index=True)
    dept_name = Column(String(100), nullable=False)
    dept_category = Column(String(100), nullable=False)
    
    # Relationships
    sales_data = relationship("SalesData", back_populates="department")
    forecasts = relationship("Forecast", back_populates="department")
    model_performance = relationship("ModelPerformance", back_populates="department")
    
    def __repr__(self):
        return f"<Department(dept_id={self.dept_id}, name='{self.dept_name}')>"

class SalesData(Base):
    """Historical sales data"""
    __tablename__ = "sales_data"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    dept_id = Column(Integer, ForeignKey("departments.dept_id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    weekly_sales = Column(Numeric(12, 2), nullable=False)
    is_holiday = Column(Boolean, default=False)
    temperature = Column(Numeric(5, 2))
    fuel_price = Column(Numeric(5, 2))
    cpi = Column(Numeric(8, 4))
    unemployment_rate = Column(Numeric(5, 2))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    store = relationship("Store", back_populates="sales_data")
    department = relationship("Department", back_populates="sales_data")
    
    def __repr__(self):
        return f"<SalesData(id={self.id}, store={self.store_id}, dept={self.dept_id}, date={self.date})>"

class Forecast(Base):
    """Sales forecasts"""
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    dept_id = Column(Integer, ForeignKey("departments.dept_id"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    forecast_horizon = Column(Integer, nullable=False)
    predicted_sales = Column(Numeric(12, 2), nullable=False)
    confidence_lower = Column(Numeric(12, 2))
    confidence_upper = Column(Numeric(12, 2))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    mape_score = Column(Numeric(8, 4))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    store = relationship("Store", back_populates="forecasts")
    department = relationship("Department", back_populates="forecasts")
    
    def __repr__(self):
        return f"<Forecast(id={self.id}, store={self.store_id}, dept={self.dept_id}, date={self.forecast_date})>"

class ModelPerformance(Base):
    """Model performance metrics"""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    dept_id = Column(Integer, ForeignKey("departments.dept_id"), nullable=False, index=True)
    mape_score = Column(Numeric(8, 4), nullable=False)
    wape_score = Column(Numeric(8, 4))
    bias_score = Column(Numeric(8, 4))
    validation_date = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    store = relationship("Store", back_populates="model_performance")
    department = relationship("Department", back_populates="model_performance")
    
    def __repr__(self):
        return f"<ModelPerformance(id={self.id}, model='{self.model_name}', version='{self.model_version}')>"

# Create composite indexes for better query performance
Index('idx_sales_store_dept_date', SalesData.store_id, SalesData.dept_id, SalesData.date)
Index('idx_forecasts_store_dept_date', Forecast.store_id, Forecast.dept_id, Forecast.forecast_date)
Index('idx_model_perf_model_version', ModelPerformance.model_name, ModelPerformance.model_version)














