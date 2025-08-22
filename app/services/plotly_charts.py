"""
Plotly.js Chart Service for Walmart Sales Forecasting Dashboard
Generates interactive Plotly.js charts for the web dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class PlotlyChartService:
    """Service for generating Plotly.js chart configurations"""
    
    def __init__(self):
        self.df = None
        self.charts = {}
        
    def load_data(self):
        """Load Walmart sales data"""
        try:
            # Try to load from data_manipulation directory
            data_path = Path("./data_manipulation/Walmart.csv")
            if not data_path.exists():
                data_path = Path("./data_manipulation/Walmart_sample.csv")
            
            if data_path.exists():
                self.df = pd.read_csv(data_path)
                self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
                print(f"Data loaded: {len(self.df):,} records, {self.df['Store'].nunique()} stores")
                return True
            else:
                print("No Walmart data file found")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def generate_sales_distribution(self):
        """Generate sales distribution histogram"""
        if self.df is None:
            return None
            
        # Calculate statistics
        mean_sales = self.df['Weekly_Sales'].mean()
        median_sales = self.df['Weekly_Sales'].median()
        
        chart_config = {
            'data': [{
                'x': self.df['Weekly_Sales'].tolist(),
                'type': 'histogram',
                'nbinsx': 40,
                'marker': {
                    'color': 'skyblue',
                    'line': {'color': 'black', 'width': 1}
                },
                'name': 'Weekly Sales'
            }],
            'layout': {
                'title': {
                    'text': 'Weekly Sales Distribution',
                    'font': {'size': 16, 'color': '#0071ce'}
                },
                'xaxis': {
                    'title': 'Weekly Sales ($)',
                    'gridcolor': '#e1e5e9'
                },
                'yaxis': {
                    'title': 'Frequency',
                    'gridcolor': '#e1e5e9'
                },
                'shapes': [{
                    'type': 'line',
                    'x0': mean_sales,
                    'x1': mean_sales,
                    'y0': 0,
                    'y1': 1,
                    'yref': 'paper',
                    'line': {'color': 'red', 'width': 2, 'dash': 'dash'},
                    'name': f'Mean: ${mean_sales:,.0f}'
                }],
                'annotations': [{
                    'x': mean_sales,
                    'y': 0.95,
                    'yref': 'paper',
                    'text': f'Mean: ${mean_sales:,.0f}',
                    'showarrow': False,
                    'font': {'color': 'red', 'size': 12}
                }],
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': False,
                'height': 400
            }
        }
        
        return chart_config
    
    def generate_store_performance(self):
        """Generate top stores bar chart"""
        if self.df is None:
            return None
            
        store_means = self.df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
        top_15_stores = store_means.head(15)
        
        chart_config = {
            'data': [{
                'x': [f'Store {store}' for store in top_15_stores.index],
                'y': top_15_stores.values.tolist(),
                'type': 'bar',
                'marker': {
                    'color': 'lightgreen',
                    'opacity': 0.7
                },
                'name': 'Average Sales'
            }],
            'layout': {
                'title': {
                    'text': 'Top 15 Stores by Average Sales',
                    'font': {'size': 16, 'color': '#0071ce'}
                },
                'xaxis': {
                    'title': 'Store',
                    'tickangle': -45,
                    'gridcolor': '#e1e5e9'
                },
                'yaxis': {
                    'title': 'Average Weekly Sales ($)',
                    'gridcolor': '#e1e5e9'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': False,
                'height': 400
            }
        }
        
        return chart_config
    
    def generate_sales_trend(self):
        """Generate sales trend over time"""
        if self.df is None:
            return None
            
        weekly_avg = self.df.groupby('Date')['Weekly_Sales'].mean()
        
        # Convert timestamps to strings for JSON serialization
        dates = [date.strftime('%Y-%m-%d') for date in weekly_avg.index]
        
        chart_config = {
            'data': [{
                'x': dates,
                'y': weekly_avg.values.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'line': {'color': 'blue', 'width': 2},
                'name': 'Average Sales'
            }],
            'layout': {
                'title': {
                    'text': 'Sales Trend Over Time (Store Average)',
                    'font': {'size': 16, 'color': '#0071ce'}
                },
                'xaxis': {
                    'title': 'Date',
                    'gridcolor': '#e1e5e9'
                },
                'yaxis': {
                    'title': 'Average Weekly Sales ($)',
                    'gridcolor': '#e1e5e9'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': False,
                'height': 400
            }
        }
        
        return chart_config
    
    def generate_holiday_effect(self):
        """Generate holiday effect box plot"""
        if self.df is None:
            return None
            
        holiday_sales = self.df[self.df['Holiday_Flag'] == 1]['Weekly_Sales']
        non_holiday_sales = self.df[self.df['Holiday_Flag'] == 0]['Weekly_Sales']
        
        chart_config = {
            'data': [
                {
                    'y': non_holiday_sales.tolist(),
                    'type': 'box',
                    'name': 'Non-Holiday',
                    'marker': {'color': 'lightblue'}
                },
                {
                    'y': holiday_sales.tolist(),
                    'type': 'box',
                    'name': 'Holiday',
                    'marker': {'color': 'orange'}
                }
            ],
            'layout': {
                'title': {
                    'text': 'Holiday Effect on Sales',
                    'font': {'size': 16, 'color': '#0071ce'}
                },
                'yaxis': {
                    'title': 'Weekly Sales ($)',
                    'gridcolor': '#e1e5e9'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': True,
                'height': 400
            }
        }
        
        return chart_config
    
    def generate_monthly_seasonality(self):
        """Generate monthly seasonality pattern"""
        if self.df is None:
            return None
            
        self.df['Month'] = self.df['Date'].dt.month
        monthly_sales = self.df.groupby('Month')['Weekly_Sales'].mean()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        chart_config = {
            'data': [{
                'x': month_names,
                'y': monthly_sales.values.tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'line': {'color': 'orange', 'width': 2},
                'marker': {'size': 8, 'color': 'orange'},
                'name': 'Monthly Sales'
            }],
            'layout': {
                'title': {
                    'text': 'Monthly Seasonality Pattern',
                    'font': {'size': 16, 'color': '#0071ce'}
                },
                'xaxis': {
                    'title': 'Month',
                    'gridcolor': '#e1e5e9'
                },
                'yaxis': {
                    'title': 'Average Weekly Sales ($)',
                    'gridcolor': '#e1e5e9'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': False,
                'height': 400
            }
        }
        
        return chart_config
    
    def generate_temperature_correlation(self):
        """Generate temperature vs sales scatter plot"""
        if self.df is None:
            return None
            
        # Calculate trend line
        z = np.polyfit(self.df['Temperature'], self.df['Weekly_Sales'], 1)
        p = np.poly1d(z)
        trend_x = np.linspace(self.df['Temperature'].min(), self.df['Temperature'].max(), 100)
        trend_y = p(trend_x)
        
        chart_config = {
            'data': [
                {
                    'x': self.df['Temperature'].tolist(),
                    'y': self.df['Weekly_Sales'].tolist(),
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {
                        'color': 'red',
                        'opacity': 0.5,
                        'size': 4
                    },
                    'name': 'Sales vs Temperature'
                },
                {
                    'x': trend_x.tolist(),
                    'y': trend_y.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'color': 'red', 'width': 2, 'dash': 'dash'},
                    'name': 'Trend Line'
                }
            ],
            'layout': {
                'title': {
                    'text': 'Temperature vs Weekly Sales',
                    'font': {'size': 16, 'color': '#0071ce'}
                },
                'xaxis': {
                    'title': 'Temperature (°F)',
                    'gridcolor': '#e1e5e9'
                },
                'yaxis': {
                    'title': 'Weekly Sales ($)',
                    'gridcolor': '#e1e5e9'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'showlegend': True,
                'height': 400
            }
        }
        
        return chart_config
    
    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON serializable"""
        if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def _clean_chart_config(self, chart_config):
        """Clean chart configuration to ensure JSON serializability"""
        if chart_config is None:
            return {}
        
        # Convert to JSON and back to ensure serializability
        try:
            json_str = json.dumps(chart_config, default=self._ensure_json_serializable)
            return json.loads(json_str)
        except Exception as e:
            print(f"Warning: Chart config cleaning failed: {e}")
            return {}

    def generate_all_charts(self):
        """Generate all charts and return as Plotly.js configurations"""
        if not self.load_data():
            return False
            
        try:
            raw_charts = {
                'sales_distribution': self.generate_sales_distribution(),
                'store_performance': self.generate_store_performance(),
                'sales_trend': self.generate_sales_trend(),
                'holiday_effect': self.generate_holiday_effect(),
                'monthly_seasonality': self.generate_monthly_seasonality(),
                'temperature_correlation': self.generate_temperature_correlation()
            }
            
            # Clean all chart configurations to ensure JSON serializability
            self.charts = {name: self._clean_chart_config(config) for name, config in raw_charts.items()}
            
            print("✅ All Plotly.js charts generated successfully!")
            return self.charts
            
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
            return False

# Create global instance
plotly_service = PlotlyChartService()
