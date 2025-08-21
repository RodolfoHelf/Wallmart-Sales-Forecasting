"""
Analytics service for generating business insights and performance metrics
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import pandas as pd
import numpy as np

from ..models.schemas import SalesAnalytics, ModelPerformanceSummary
from ..models.database_models import (
    SalesData, Forecast, ModelPerformance, Store, Department
)

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for generating analytics and insights"""
    
    async def get_sales_analytics(
        self,
        store_id: Optional[int],
        dept_id: Optional[int],
        start_date: Optional[date],
        end_date: Optional[date],
        db: Session
    ) -> Dict[str, Any]:
        """Get comprehensive sales analytics and trends"""
        try:
            # Build base query
            query = db.query(SalesData)
            
            if store_id:
                query = query.filter(SalesData.store_id == store_id)
            if dept_id:
                query = query.filter(SalesData.dept_id == dept_id)
            if start_date:
                query = query.filter(SalesData.date >= start_date)
            if end_date:
                query = query.filter(SalesData.date <= end_date)
            
            # Get sales data
            sales_data = query.order_by(SalesData.date).all()
            
            if not sales_data:
                return {
                    'total_sales': 0,
                    'avg_weekly_sales': 0,
                    'sales_trend': 'no_data',
                    'seasonal_pattern': None,
                    'holiday_impact': None,
                    'top_performing_stores': [],
                    'top_performing_departments': []
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([
                {
                    'date': s.date,
                    'weekly_sales': float(s.weekly_sales),
                    'is_holiday': s.is_holiday,
                    'store_id': s.store_id,
                    'dept_id': s.dept_id
                }
                for s in sales_data
            ])
            
            # Calculate basic metrics
            total_sales = df['weekly_sales'].sum()
            avg_weekly_sales = df['weekly_sales'].mean()
            
            # Determine sales trend
            if len(df) >= 2:
                recent_sales = df.tail(4)['weekly_sales'].mean()
                earlier_sales = df.head(4)['weekly_sales'].mean()
                
                if recent_sales > earlier_sales * 1.05:
                    sales_trend = 'increasing'
                elif recent_sales < earlier_sales * 0.95:
                    sales_trend = 'decreasing'
                else:
                    sales_trend = 'stable'
            else:
                sales_trend = 'insufficient_data'
            
            # Analyze seasonal patterns
            seasonal_pattern = self._analyze_seasonality(df)
            
            # Calculate holiday impact
            holiday_impact = self._calculate_holiday_impact(df)
            
            # Get top performing stores and departments
            top_stores = await self._get_top_performing_stores(db, store_id, dept_id, start_date, end_date)
            top_departments = await self._get_top_performing_departments(db, store_id, dept_id, start_date, end_date)
            
            return {
                'total_sales': total_sales,
                'avg_weekly_sales': avg_weekly_sales,
                'sales_trend': sales_trend,
                'seasonal_pattern': seasonal_pattern,
                'holiday_impact': holiday_impact,
                'top_performing_stores': top_stores,
                'top_performing_departments': top_departments
            }
            
        except Exception as e:
            logger.error(f"Error generating sales analytics: {str(e)}")
            raise
    
    async def get_model_performance(
        self,
        model_name: Optional[str],
        store_id: Optional[int],
        dept_id: Optional[int],
        db: Session
    ) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            # Build base query
            query = db.query(ModelPerformance)
            
            if model_name:
                query = query.filter(ModelPerformance.model_name == model_name)
            if store_id:
                query = query.filter(ModelPerformance.store_id == store_id)
            if dept_id:
                query = query.filter(ModelPerformance.dept_id == dept_id)
            
            # Get performance data
            performance_data = query.all()
            
            if not performance_data:
                return {
                    'overall_metrics': {
                        'mape': 0,
                        'wape': 0,
                        'bias': 0
                    },
                    'model_breakdown': [],
                    'store_performance': [],
                    'department_performance': []
                }
            
            # Calculate overall metrics
            overall_mape = np.mean([float(p.mape_score) for p in performance_data])
            overall_wape = np.mean([float(p.wape_score) for p in performance_data if p.wape_score])
            overall_bias = np.mean([float(p.bias_score) for p in performance_data if p.bias_score])
            
            # Model breakdown
            model_breakdown = {}
            for p in performance_data:
                model_key = f"{p.model_name}_{p.model_version}"
                if model_key not in model_breakdown:
                    model_breakdown[model_key] = {
                        'model_name': p.model_name,
                        'model_version': p.model_version,
                        'count': 0,
                        'avg_mape': 0,
                        'avg_wape': 0,
                        'avg_bias': 0
                    }
                
                model_breakdown[model_key]['count'] += 1
                model_breakdown[model_key]['avg_mape'] += float(p.mape_score)
                if p.wape_score:
                    model_breakdown[model_key]['avg_wape'] += float(p.wape_score)
                if p.bias_score:
                    model_breakdown[model_key]['avg_bias'] += float(p.bias_score)
            
            # Calculate averages
            for model_key in model_breakdown:
                count = model_breakdown[model_key]['count']
                model_breakdown[model_key]['avg_mape'] /= count
                if model_breakdown[model_key]['avg_wape'] > 0:
                    model_breakdown[model_key]['avg_wape'] /= count
                if model_breakdown[model_key]['avg_bias'] > 0:
                    model_breakdown[model_key]['avg_bias'] /= count
            
            # Store and department performance
            store_performance = await self._get_store_model_performance(db, model_name, store_id, dept_id)
            dept_performance = await self._get_department_model_performance(db, model_name, store_id, dept_id)
            
            return {
                'overall_metrics': {
                    'mape': overall_mape,
                    'wape': overall_wape,
                    'bias': overall_bias
                },
                'model_breakdown': list(model_breakdown.values()),
                'store_performance': store_performance,
                'department_performance': dept_performance
            }
            
        except Exception as e:
            logger.error(f"Error retrieving model performance: {str(e)}")
            raise
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Optional[str]:
        """Analyze seasonal patterns in sales data"""
        try:
            if len(df) < 52:  # Need at least a year of data
                return None
            
            # Add month column
            df['month'] = pd.to_datetime(df['date']).dt.month
            
            # Calculate monthly averages
            monthly_avg = df.groupby('month')['weekly_sales'].mean()
            
            # Find peak and trough months
            peak_month = monthly_avg.idxmax()
            trough_month = monthly_avg.idxmin()
            
            # Determine seasonality type
            if peak_month in [11, 12] and trough_month in [1, 2]:
                return "holiday_peak_winter_trough"
            elif peak_month in [6, 7, 8] and trough_month in [1, 2]:
                return "summer_peak_winter_trough"
            elif monthly_avg.max() / monthly_avg.min() > 1.5:
                return "strong_seasonal"
            elif monthly_avg.max() / monthly_avg.min() > 1.2:
                return "moderate_seasonal"
            else:
                return "low_seasonal"
                
        except Exception as e:
            logger.warning(f"Error analyzing seasonality: {str(e)}")
            return None
    
    def _calculate_holiday_impact(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate the impact of holidays on sales"""
        try:
            if 'is_holiday' not in df.columns:
                return None
            
            holiday_sales = df[df['is_holiday']]['weekly_sales']
            non_holiday_sales = df[~df['is_holiday']]['weekly_sales']
            
            if len(holiday_sales) == 0 or len(non_holiday_sales) == 0:
                return None
            
            holiday_avg = holiday_sales.mean()
            non_holiday_avg = non_holiday_sales.mean()
            
            if non_holiday_avg == 0:
                return None
            
            impact = ((holiday_avg - non_holiday_avg) / non_holiday_avg) * 100
            return float(impact)
            
        except Exception as e:
            logger.warning(f"Error calculating holiday impact: {str(e)}")
            return None
    
    async def _get_top_performing_stores(
        self, 
        db: Session, 
        store_id: Optional[int], 
        dept_id: Optional[int], 
        start_date: Optional[date], 
        end_date: Optional[date]
    ) -> List[Dict[str, Any]]:
        """Get top performing stores"""
        try:
            query = db.query(
                Store.store_id,
                Store.store_name,
                Store.location_state,
                func.sum(SalesData.weekly_sales).label('total_sales'),
                func.avg(SalesData.weekly_sales).label('avg_weekly_sales')
            ).join(SalesData)
            
            if store_id:
                query = query.filter(Store.store_id == store_id)
            if dept_id:
                query = query.filter(SalesData.dept_id == dept_id)
            if start_date:
                query = query.filter(SalesData.date >= start_date)
            if end_date:
                query = query.filter(SalesData.date <= end_date)
            
            top_stores = query.group_by(
                Store.store_id, Store.store_name, Store.location_state
            ).order_by(desc(func.sum(SalesData.weekly_sales))).limit(5).all()
            
            return [
                {
                    'store_id': s.store_id,
                    'store_name': s.store_name,
                    'location_state': s.location_state,
                    'total_sales': float(s.total_sales),
                    'avg_weekly_sales': float(s.avg_weekly_sales)
                }
                for s in top_stores
            ]
            
        except Exception as e:
            logger.error(f"Error getting top performing stores: {str(e)}")
            return []
    
    async def _get_top_performing_departments(
        self, 
        db: Session, 
        store_id: Optional[int], 
        dept_id: Optional[int], 
        start_date: Optional[date], 
        end_date: Optional[date]
    ) -> List[Dict[str, Any]]:
        """Get top performing departments"""
        try:
            query = db.query(
                Department.dept_id,
                Department.dept_name,
                Department.dept_category,
                func.sum(SalesData.weekly_sales).label('total_sales'),
                func.avg(SalesData.weekly_sales).label('avg_weekly_sales')
            ).join(SalesData)
            
            if store_id:
                query = query.filter(SalesData.store_id == store_id)
            if dept_id:
                query = query.filter(Department.dept_id == dept_id)
            if start_date:
                query = query.filter(SalesData.date >= start_date)
            if end_date:
                query = query.filter(SalesData.date <= end_date)
            
            top_departments = query.group_by(
                Department.dept_id, Department.dept_name, Department.dept_category
            ).order_by(desc(func.sum(SalesData.weekly_sales))).limit(5).all()
            
            return [
                {
                    'dept_id': d.dept_id,
                    'dept_name': d.dept_name,
                    'dept_category': d.dept_category,
                    'total_sales': float(d.total_sales),
                    'avg_weekly_sales': float(d.avg_weekly_sales)
                }
                for d in top_departments
            ]
            
        except Exception as e:
            logger.error(f"Error getting top performing departments: {str(e)}")
            return []
    
    async def _get_store_model_performance(
        self, 
        db: Session, 
        model_name: Optional[str], 
        store_id: Optional[int], 
        dept_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get model performance by store"""
        try:
            query = db.query(
                Store.store_id,
                Store.store_name,
                func.avg(ModelPerformance.mape_score).label('avg_mape'),
                func.count(ModelPerformance.id).label('num_models')
            ).join(ModelPerformance)
            
            if model_name:
                query = query.filter(ModelPerformance.model_name == model_name)
            if store_id:
                query = query.filter(Store.store_id == store_id)
            if dept_id:
                query = query.filter(ModelPerformance.dept_id == dept_id)
            
            store_performance = query.group_by(
                Store.store_id, Store.store_name
            ).order_by(func.avg(ModelPerformance.mape_score)).limit(10).all()
            
            return [
                {
                    'store_id': s.store_id,
                    'store_name': s.store_name,
                    'avg_mape': float(s.avg_mape),
                    'num_models': s.num_models
                }
                for s in store_performance
            ]
            
        except Exception as e:
            logger.error(f"Error getting store model performance: {str(e)}")
            return []
    
    async def _get_department_model_performance(
        self, 
        db: Session, 
        model_name: Optional[str], 
        store_id: Optional[int], 
        dept_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get model performance by department"""
        try:
            query = db.query(
                Department.dept_id,
                Department.dept_name,
                func.avg(ModelPerformance.mape_score).label('avg_mape'),
                func.count(ModelPerformance.id).label('num_models')
            ).join(ModelPerformance)
            
            if model_name:
                query = query.filter(ModelPerformance.model_name == model_name)
            if store_id:
                query = query.filter(ModelPerformance.store_id == store_id)
            if dept_id:
                query = query.filter(Department.dept_id == dept_id)
            
            dept_performance = query.group_by(
                Department.dept_id, Department.dept_name
            ).order_by(func.avg(ModelPerformance.mape_score)).limit(10).all()
            
            return [
                {
                    'dept_id': d.dept_id,
                    'dept_name': d.dept_name,
                    'avg_mape': float(d.avg_mape),
                    'num_models': d.num_models
                }
                for d in dept_performance
            ]
            
        except Exception as e:
            logger.error(f"Error getting department model performance: {str(e)}")
            return []














