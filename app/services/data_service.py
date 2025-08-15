"""
Data service for handling sales data operations
"""

from typing import List, Optional
from datetime import date
from decimal import Decimal
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from ..models.schemas import SalesData, Store, Department
from ..models.database_models import SalesData as SalesDataModel, Store as StoreModel, Department as DepartmentModel

logger = logging.getLogger(__name__)

class DataService:
    """Service for handling data operations"""
    
    async def get_sales_data(
        self,
        store_id: Optional[int],
        dept_id: Optional[int],
        start_date: Optional[date],
        end_date: Optional[date],
        limit: int,
        db: Session
    ) -> List[SalesData]:
        """Retrieve sales data with optional filtering"""
        try:
            query = db.query(SalesDataModel)
            
            if store_id:
                query = query.filter(SalesDataModel.store_id == store_id)
            if dept_id:
                query = query.filter(SalesDataModel.dept_id == dept_id)
            if start_date:
                query = query.filter(SalesDataModel.date >= start_date)
            if end_date:
                query = query.filter(SalesDataModel.date <= end_date)
            
            sales_data = query.order_by(desc(SalesDataModel.date)).limit(limit).all()
            
            return [
                SalesData(
                    id=s.id,
                    store_id=s.store_id,
                    dept_id=s.dept_id,
                    date=s.date,
                    weekly_sales=s.weekly_sales,
                    is_holiday=s.is_holiday,
                    temperature=s.temperature,
                    fuel_price=s.fuel_price,
                    cpi=s.cpi,
                    unemployment_rate=s.unemployment_rate,
                    created_at=s.created_at
                )
                for s in sales_data
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving sales data: {str(e)}")
            raise
    
    async def get_stores(self, db: Session) -> List[Store]:
        """Get all stores"""
        try:
            stores = db.query(StoreModel).all()
            
            return [
                Store(
                    store_id=s.store_id,
                    store_name=s.store_name,
                    store_type=s.store_type,
                    store_size_sqft=s.store_size_sqft,
                    location_state=s.location_state,
                    location_city=s.location_city
                )
                for s in stores
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving stores: {str(e)}")
            raise
    
    async def get_departments(self, db: Session) -> List[Department]:
        """Get all departments"""
        try:
            departments = db.query(DepartmentModel).all()
            
            return [
                Department(
                    dept_id=d.dept_id,
                    dept_name=d.dept_name,
                    dept_category=d.dept_category
                )
                for d in departments
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving departments: {str(e)}")
            raise
    
    async def get_sales_summary(self, db: Session) -> dict:
        """Get sales data summary statistics"""
        try:
            # Total sales records
            total_records = db.query(func.count(SalesDataModel.id)).scalar()
            
            # Total sales amount
            total_sales = db.query(func.sum(SalesDataModel.weekly_sales)).scalar() or Decimal('0')
            
            # Average weekly sales
            avg_sales = db.query(func.avg(SalesDataModel.weekly_sales)).scalar() or Decimal('0')
            
            # Date range
            date_range = db.query(
                func.min(SalesDataModel.date),
                func.max(SalesDataModel.date)
            ).first()
            
            return {
                'total_records': total_records,
                'total_sales': total_sales,
                'avg_weekly_sales': avg_sales,
                'date_range': {
                    'start': date_range[0] if date_range[0] else None,
                    'end': date_range[1] if date_range[1] else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving sales summary: {str(e)}")
            raise
    
    async def get_store_performance(self, db: Session, limit: int = 10) -> List[dict]:
        """Get top performing stores by sales"""
        try:
            store_performance = db.query(
                StoreModel.store_id,
                StoreModel.store_name,
                StoreModel.location_state,
                func.sum(SalesDataModel.weekly_sales).label('total_sales'),
                func.avg(SalesDataModel.weekly_sales).label('avg_weekly_sales'),
                func.count(SalesDataModel.id).label('num_records')
            ).join(SalesDataModel).group_by(
                StoreModel.store_id,
                StoreModel.store_name,
                StoreModel.location_state
            ).order_by(desc(func.sum(SalesDataModel.weekly_sales))).limit(limit).all()
            
            return [
                {
                    'store_id': p.store_id,
                    'store_name': p.store_name,
                    'location_state': p.location_state,
                    'total_sales': float(p.total_sales),
                    'avg_weekly_sales': float(p.avg_weekly_sales),
                    'num_records': p.num_records
                }
                for p in store_performance
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving store performance: {str(e)}")
            raise
    
    async def get_department_performance(self, db: Session, limit: int = 10) -> List[dict]:
        """Get top performing departments by sales"""
        try:
            dept_performance = db.query(
                DepartmentModel.dept_id,
                DepartmentModel.dept_name,
                DepartmentModel.dept_category,
                func.sum(SalesDataModel.weekly_sales).label('total_sales'),
                func.avg(SalesDataModel.weekly_sales).label('avg_weekly_sales'),
                func.count(SalesDataModel.id).label('num_records')
            ).join(SalesDataModel).group_by(
                DepartmentModel.dept_id,
                DepartmentModel.dept_name,
                DepartmentModel.dept_category
            ).order_by(desc(func.sum(SalesDataModel.weekly_sales))).limit(limit).all()
            
            return [
                {
                    'dept_id': p.dept_id,
                    'dept_name': p.dept_name,
                    'dept_category': p.dept_category,
                    'total_sales': float(p.total_sales),
                    'avg_weekly_sales': float(p.avg_weekly_sales),
                    'num_records': p.num_records
                }
                for p in dept_performance
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving department performance: {str(e)}")
            raise










