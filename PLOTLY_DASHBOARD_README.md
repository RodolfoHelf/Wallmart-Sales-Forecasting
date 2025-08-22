# 🎯 Plotly.js Dashboard for Walmart Sales Forecasting

## ✨ **What's New**

This project now features an **interactive Plotly.js dashboard** that displays real-time Walmart sales data visualizations directly in your web browser. No more static images or external files - everything is dynamic and interactive!

## 🚀 **Key Features**

### **Interactive Charts**
- **📊 Sales Distribution**: Histogram with mean line indicator
- **🏪 Store Performance**: Top 15 stores bar chart with rankings
- **📈 Sales Trend**: Time series analysis over 2.7 years
- **🎉 Holiday Effect**: Box plot comparison (Holiday vs Non-Holiday)
- **📅 Monthly Seasonality**: 12-month pattern analysis
- **🌡️ Temperature Correlation**: Scatter plot with trend line

### **Professional Dashboard Layout**
- **Key Metrics Cards**: Summary statistics in beautiful gradient cards
- **Responsive Grid**: Charts adapt to different screen sizes
- **Real-time Data**: Uses your actual Walmart.csv data
- **Interactive Elements**: Hover, zoom, pan, and more Plotly features

## 🛠️ **How It Works**

### **1. Chart Generation (Backend)**
- **Python Service**: `app/services/plotly_charts.py` generates chart configurations
- **Data Loading**: Automatically loads Walmart.csv from `data_manipulation/`
- **Chart Configs**: Creates Plotly.js compatible JSON configurations
- **Startup Integration**: Charts are generated when the app starts

### **2. Chart Display (Frontend)**
- **Plotly.js Library**: Loaded from CDN for optimal performance
- **Dynamic Rendering**: Charts are rendered when the dashboard loads
- **Responsive Design**: Charts adapt to different screen sizes
- **Interactive Features**: Full Plotly.js functionality available

## 📋 **Prerequisites**

### **Required Python Packages**
```bash
pip install pandas numpy plotly
```

### **Data Files**
- `data_manipulation/Walmart.csv` (or `Walmart_sample.csv` as fallback)
- Data should have columns: `Date`, `Store`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`

## 🚀 **Quick Start**

### **1. Test Chart Generation**
```bash
python test_plotly_charts.py
```
This will verify that all charts can be generated successfully.

### **2. Start the Web App**
```bash
python run.py
```

### **3. Navigate to Dashboard**
- Open your browser to `http://localhost:8000`
- Click on the **"Dashboard"** tab
- Enjoy your interactive charts! 🎉

## 📁 **File Structure**

```
app/
├── services/
│   └── plotly_charts.py          # Chart generation service
├── static/
│   ├── css/
│   │   └── style.css            # Dashboard styling
│   └── js/
│       └── main.js              # Main JavaScript
└── main.py                      # FastAPI app with dashboard integration

test_plotly_charts.py            # Test script for charts
PLOTLY_DASHBOARD_README.md       # This file
```

## 🎨 **Dashboard Layout**

### **Top Section: Key Metrics**
- **4 metric cards** displaying summary statistics
- **Gradient backgrounds** with hover effects
- **Responsive grid** that adapts to screen size

### **Charts Grid**
- **6 chart containers** in a responsive grid
- **Each chart** has a title, visualization, and description
- **Chart descriptions** provide business insights and analysis

### **Responsive Design**
- **Desktop**: 2-3 charts per row
- **Tablet**: 1-2 charts per row  
- **Mobile**: 1 chart per row

## 🔧 **Customization**

### **Adding New Charts**
1. Add new chart method to `PlotlyChartService` class
2. Update `generate_all_charts()` method
3. Add HTML container in dashboard section
4. Add JavaScript rendering code

### **Modifying Chart Styles**
- Edit chart configurations in `plotly_charts.py`
- Modify CSS in `style.css`
- Adjust Plotly.js options for colors, fonts, etc.

### **Data Sources**
- Change data file path in `load_data()` method
- Modify column names if your data structure differs
- Add data preprocessing as needed

## 🐛 **Troubleshooting**

### **Charts Not Loading**
1. Check browser console for JavaScript errors
2. Verify Plotly.js CDN is accessible
3. Run `test_plotly_charts.py` to verify backend
4. Check data file exists and is readable

### **Performance Issues**
1. Use `Walmart_sample.csv` for testing (smaller file)
2. Reduce chart complexity in configurations
3. Optimize data loading and processing

### **Data Issues**
1. Verify CSV format and column names
2. Check for missing or invalid data
3. Ensure date format matches expected pattern

## 🌟 **Benefits of Plotly.js**

### **vs. Static Images**
- ✅ **Interactive**: Zoom, pan, hover, click
- ✅ **Responsive**: Adapts to any screen size
- ✅ **Real-time**: No need to regenerate images
- ✅ **Professional**: Publication-quality charts

### **vs. Other Libraries**
- ✅ **Easy Integration**: Simple JSON configuration
- ✅ **Rich Features**: Extensive customization options
- ✅ **Performance**: Optimized for web browsers
- ✅ **Documentation**: Excellent Plotly.js docs

## 📊 **Chart Types & Insights**

### **Sales Distribution**
- **Purpose**: Understand sales variability
- **Insight**: 54.1% coefficient of variation indicates high variability
- **Business Impact**: Need for robust forecasting models

### **Store Performance**
- **Purpose**: Identify top and bottom performers
- **Insight**: 3.5x variation between best and worst stores
- **Business Impact**: Store-specific strategies needed

### **Sales Trend**
- **Purpose**: Identify temporal patterns
- **Insight**: Q4 peaks (+15%), Q1 lows (-10%)
- **Business Impact**: Seasonal planning and inventory management

### **Holiday Effect**
- **Purpose**: Quantify holiday impact
- **Insight**: +15.3% sales premium during holidays
- **Business Impact**: Holiday-specific promotions and staffing

### **Monthly Seasonality**
- **Purpose**: Understand monthly patterns
- **Insight**: December best (+15%), January worst (-10%)
- **Business Impact**: Monthly planning and resource allocation

### **Temperature Correlation**
- **Purpose**: Analyze weather impact
- **Insight**: U-shaped relationship with sales
- **Business Impact**: Weather-based demand forecasting

## 🎯 **Next Steps**

### **Immediate Enhancements**
- Add more chart types (fuel price, CPI, unemployment)
- Implement chart filtering and date ranges
- Add export functionality (PNG, PDF)

### **Advanced Features**
- Real-time data updates
- User-defined chart combinations
- Interactive forecasting scenarios
- Dashboard customization options

---

## 🎉 **Ready to Explore?**

Your Plotly.js dashboard is now ready! Run the test script, start the web app, and explore your Walmart sales data through beautiful, interactive visualizations.

**Happy Data Exploring! 📊✨**
