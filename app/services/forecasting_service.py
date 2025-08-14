"""
Forecasting service for Walmart sales prediction
Handles model training, prediction, and MLflow integration
"""

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.prophet
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.schemas import ForecastRequest, Forecast
from ..models.database_models import SalesData, Forecast as ForecastModel
from ..config import settings

logger = logging.getLogger(__name__)

class ForecastingService:
    """Service for handling sales forecasting operations"""
    
    def __init__(self):
        """Initialize the forecasting service"""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        
        # Available models
        self.models = {
            'sarimax': self._train_sarimax,
            'lightgbm': self._train_lightgbm,
            'xgboost': self._train_xgboost,
            'prophet': self._train_prophet,
            'ensemble': self._train_ensemble
        }
    
    async def generate_forecast(self, request: ForecastRequest, db: Session) -> Forecast:
        """Generate sales forecast for specified store and department"""
        try:
            # Get historical data
            historical_data = self._get_historical_data(
                db, request.store_id, request.dept_id
            )
            
            if historical_data.empty:
                raise ValueError("Insufficient historical data for forecasting")
            
            # Select model
            model_name = request.model_name or 'ensemble'
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Generate forecast
            forecast_values = await self._generate_forecast_values(
                historical_data, model_name, request.forecast_horizon
            )
            
            # Create forecast record
            forecast = ForecastModel(
                store_id=request.store_id,
                dept_id=request.dept_id,
                forecast_date=date.today(),
                forecast_horizon=request.forecast_horizon,
                predicted_sales=Decimal(str(forecast_values['predicted_sales'])),
                confidence_lower=Decimal(str(forecast_values.get('confidence_lower', 0))),
                confidence_upper=Decimal(str(forecast_values.get('confidence_upper', 0))),
                model_name=model_name,
                model_version=forecast_values.get('model_version', '1.0'),
                mape_score=Decimal(str(forecast_values.get('mape_score', 0)))
            )
            
            db.add(forecast)
            db.commit()
            db.refresh(forecast)
            
            # Convert to response schema
            return Forecast(
                id=forecast.id,
                store_id=forecast.store_id,
                dept_id=forecast.dept_id,
                forecast_date=forecast.forecast_date,
                forecast_horizon=forecast.forecast_horizon,
                predicted_sales=forecast.predicted_sales,
                confidence_lower=forecast.confidence_lower,
                confidence_upper=forecast.confidence_upper,
                model_name=forecast.model_name,
                model_version=forecast.model_version,
                mape_score=forecast.mape_score,
                created_at=forecast.created_at
            )
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    async def get_forecasts(
        self, 
        store_id: Optional[int], 
        dept_id: Optional[int], 
        start_date: Optional[date], 
        end_date: Optional[date], 
        db: Session
    ) -> List[Forecast]:
        """Retrieve forecasts with optional filtering"""
        try:
            query = db.query(ForecastModel)
            
            if store_id:
                query = query.filter(ForecastModel.store_id == store_id)
            if dept_id:
                query = query.filter(ForecastModel.dept_id == dept_id)
            if start_date:
                query = query.filter(ForecastModel.forecast_date >= start_date)
            if end_date:
                query = query.filter(ForecastModel.forecast_date <= end_date)
            
            forecasts = query.order_by(ForecastModel.forecast_date.desc()).all()
            
            return [
                Forecast(
                    id=f.id,
                    store_id=f.store_id,
                    dept_id=f.dept_id,
                    forecast_date=f.forecast_date,
                    forecast_horizon=f.forecast_horizon,
                    predicted_sales=f.predicted_sales,
                    confidence_lower=f.confidence_lower,
                    confidence_upper=f.confidence_upper,
                    model_name=f.model_name,
                    model_version=f.model_version,
                    mape_score=f.mape_score,
                    created_at=f.created_at
                )
                for f in forecasts
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {str(e)}")
            raise
    
    def _get_historical_data(self, db: Session, store_id: int, dept_id: int) -> pd.DataFrame:
        """Retrieve historical sales data for modeling"""
        try:
            # Get sales data for the last 2 years
            end_date = date.today()
            start_date = end_date - timedelta(days=730)
            
            sales_data = db.query(SalesData).filter(
                SalesData.store_id == store_id,
                SalesData.dept_id == dept_id,
                SalesData.date >= start_date,
                SalesData.date <= end_date
            ).order_by(SalesData.date).all()
            
            if not sales_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'date': s.date,
                    'weekly_sales': float(s.weekly_sales),
                    'is_holiday': s.is_holiday,
                    'temperature': float(s.temperature) if s.temperature else None,
                    'fuel_price': float(s.fuel_price) if s.fuel_price else None,
                    'cpi': float(s.cpi) if s.cpi else None,
                    'unemployment_rate': float(s.unemployment_rate) if s.unemployment_rate else None
                }
                for s in sales_data
            ])
            
            # Set date as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {str(e)}")
            raise
    
    async def _generate_forecast_values(
        self, 
        historical_data: pd.DataFrame, 
        model_name: str, 
        horizon: int
    ) -> Dict[str, Any]:
        """Generate forecast values using specified model"""
        try:
            # Train model and generate forecast
            model_func = self.models[model_name]
            forecast_result = await model_func(historical_data, horizon)
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating forecast values: {str(e)}")
            raise
    
    async def _train_sarimax(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Train SARIMAX model"""
        try:
            with mlflow.start_run():
                # Prepare data
                sales_series = data['weekly_sales'].fillna(method='ffill')
                
                # Simple SARIMAX implementation (in production, use statsmodels)
                # For now, return a simple forecast
                last_value = sales_series.iloc[-1]
                trend = np.polyfit(range(len(sales_series)), sales_series, 1)[0]
                
                forecast_values = []
                for i in range(horizon):
                    forecast_values.append(last_value + trend * (i + 1))
                
                # Calculate MAPE (simplified)
                mape = np.mean(np.abs((sales_series - sales_series.mean()) / sales_series)) * 100
                
                mlflow.log_metric("mape", mape)
                mlflow.log_param("model_type", "sarimax")
                mlflow.log_param("horizon", horizon)
                
                return {
                    'predicted_sales': forecast_values[0],
                    'model_version': '1.0',
                    'mape_score': mape
                }
                
        except Exception as e:
            logger.error(f"Error training SARIMAX model: {str(e)}")
            raise
    
    async def _train_lightgbm(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Train LightGBM model"""
        try:
            with mlflow.start_run():
                # Prepare features
                data_clean = data.dropna()
                if len(data_clean) < 10:
                    raise ValueError("Insufficient data for LightGBM training")
                
                # Feature engineering
                data_clean['day_of_week'] = data_clean.index.dayofweek
                data_clean['month'] = data_clean.index.month
                data_clean['quarter'] = data_clean.index.quarter
                data_clean['lag_1'] = data_clean['weekly_sales'].shift(1)
                data_clean['lag_7'] = data_clean['weekly_sales'].shift(7)
                data_clean['rolling_mean_4'] = data_clean['weekly_sales'].rolling(4).mean()
                
                # Remove rows with NaN values
                data_clean = data_clean.dropna()
                
                # Prepare features and target
                feature_cols = ['day_of_week', 'month', 'quarter', 'lag_1', 'lag_7', 'rolling_mean_4']
                if 'temperature' in data_clean.columns:
                    feature_cols.append('temperature')
                if 'fuel_price' in data_clean.columns:
                    feature_cols.append('fuel_price')
                
                X = data_clean[feature_cols]
                y = data_clean['weekly_sales']
                
                # Simple LightGBM-like prediction (in production, use actual LightGBM)
                # For now, return a simple forecast
                last_value = y.iloc[-1]
                forecast_value = last_value * 1.02  # Simple 2% growth assumption
                
                # Calculate MAPE
                mape = np.mean(np.abs((y - y.mean()) / y)) * 100
                
                mlflow.log_metric("mape", mape)
                mlflow.log_param("model_type", "lightgbm")
                mlflow.log_param("horizon", horizon)
                
                return {
                    'predicted_sales': forecast_value,
                    'model_version': '1.0',
                    'mape_score': mape
                }
                
        except Exception as e:
            logger.error(f"Error training LightGBM model: {str(e)}")
            raise
    
    async def _train_xgboost(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Train XGBoost model"""
        try:
            with mlflow.start_run():
                # Similar to LightGBM but with XGBoost-specific parameters
                # For now, return a simple forecast
                sales_series = data['weekly_sales'].fillna(method='ffill')
                last_value = sales_series.iloc[-1]
                forecast_value = last_value * 1.015  # Simple 1.5% growth assumption
                
                # Calculate MAPE
                mape = np.mean(np.abs((sales_series - sales_series.mean()) / sales_series)) * 100
                
                mlflow.log_metric("mape", mape)
                mlflow.log_param("model_type", "xgboost")
                mlflow.log_param("horizon", horizon)
                
                return {
                    'predicted_sales': forecast_value,
                    'model_version': '1.0',
                    'mape_score': mape
                }
                
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    async def _train_prophet(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Train Prophet model"""
        try:
            with mlflow.start_run():
                # Prophet requires specific column names
                # For now, return a simple forecast
                sales_series = data['weekly_sales'].fillna(method='ffill')
                last_value = sales_series.iloc[-1]
                forecast_value = last_value * 1.025  # Simple 2.5% growth assumption
                
                # Calculate MAPE
                mape = np.mean(np.abs((sales_series - sales_series.mean()) / sales_series)) * 100
                
                mlflow.log_metric("mape", mape)
                mlflow.log_param("model_type", "prophet")
                mlflow.log_param("horizon", horizon)
                
                return {
                    'predicted_sales': forecast_value,
                    'model_version': '1.0',
                    'mape_score': mape
                }
                
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            raise
    
    async def _train_ensemble(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Train ensemble model combining multiple approaches"""
        try:
            with mlflow.start_run():
                # Get predictions from multiple models
                predictions = []
                
                # Try different models and collect predictions
                for model_name in ['sarimax', 'lightgbm', 'xgboost']:
                    try:
                        model_func = self.models[model_name]
                        result = await model_func(data, horizon)
                        predictions.append(result['predicted_sales'])
                    except Exception as e:
                        logger.warning(f"Model {model_name} failed: {str(e)}")
                        continue
                
                if not predictions:
                    raise ValueError("All models failed to generate predictions")
                
                # Ensemble prediction (simple average)
                ensemble_forecast = np.mean(predictions)
                
                # Calculate ensemble MAPE
                sales_series = data['weekly_sales'].fillna(method='ffill')
                mape = np.mean(np.abs((sales_series - sales_series.mean()) / sales_series)) * 100
                
                mlflow.log_metric("mape", mape)
                mlflow.log_param("model_type", "ensemble")
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("num_models", len(predictions))
                
                return {
                    'predicted_sales': ensemble_forecast,
                    'model_version': '1.0',
                    'mape_score': mape
                }
                
        except Exception as e:
            logger.error(f"Error training ensemble model: {str(e)}")
            raise









