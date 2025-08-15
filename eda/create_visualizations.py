# Walmart Sales Forecasting - Comprehensive Data Visualizations
# This script creates all the key visualizations to showcase data behavior and patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style and parameters for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data():
    """Load and prepare the Walmart dataset for visualization"""
    # Load data
    df = pd.read_csv('data/Walmart.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Add time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Add temperature categories
    df['Temp_Category'] = pd.cut(df['Temperature'], 
                                 bins=[-10, 32, 70, 90, 110], 
                                 labels=['Very Cold', 'Cold', 'Moderate', 'Hot'])
    
    # Add sales categories
    df['Sales_Category'] = pd.cut(df['Weekly_Sales'], 
                                  bins=[0, 500000, 1000000, 2000000, 7000000], 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

def create_overview_visualizations(df):
    """Create overview visualizations showing dataset structure and basic patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Walmart Sales Dataset Overview', fontsize=16, fontweight='bold')
    
    # 1. Dataset Timeline
    weekly_counts = df.groupby('Date')['Store'].count()
    axes[0, 0].plot(weekly_counts.index, weekly_counts.values, linewidth=2, color='blue')
    axes[0, 0].set_title('Weekly Data Coverage (All Stores)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Stores')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Store Distribution
    store_counts = df['Store'].value_counts().sort_index()
    axes[0, 1].bar(store_counts.index, store_counts.values, color='green', alpha=0.7)
    axes[0, 1].set_title('Data Points per Store')
    axes[0, 1].set_xlabel('Store Number')
    axes[0, 1].set_ylabel('Number of Weeks')
    
    # 3. Holiday Distribution Over Time
    holiday_weekly = df.groupby('Date')['Holiday_Flag'].sum()
    axes[1, 0].plot(holiday_weekly.index, holiday_weekly.values, linewidth=2, color='red', marker='o')
    axes[1, 0].set_title('Holiday Weeks Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Number of Holiday Stores')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Data Completeness Heatmap
    pivot_data = df.pivot_table(index='Store', columns='Year', values='Weekly_Sales', aggfunc='count')
    sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Data Completeness by Store and Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Store Number')
    
    plt.tight_layout()
    plt.savefig('walmart_overview_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_target_analysis_visualizations(df):
    """Create visualizations for the target variable (Weekly_Sales)"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Weekly Sales Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sales Distribution
    axes[0, 0].hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Weekly Sales Distribution')
    axes[0, 0].set_xlabel('Weekly Sales ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Sales Distribution by Store
    store_means = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(store_means)), store_means.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Average Sales by Store')
    axes[0, 1].set_xlabel('Store Number (Ranked)')
    axes[0, 1].set_ylabel('Average Weekly Sales ($)')
    
    # 3. Sales Box Plot by Store (Top 10)
    top_stores = store_means.head(10).index
    top_store_data = df[df['Store'].isin(top_stores)]
    sns.boxplot(data=top_store_data, x='Store', y='Weekly_Sales', ax=axes[0, 2])
    axes[0, 2].set_title('Sales Distribution - Top 10 Stores')
    axes[0, 2].set_xlabel('Store Number')
    axes[0, 2].set_ylabel('Weekly Sales ($)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Sales by Holiday vs Non-Holiday
    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']
    
    axes[1, 0].boxplot([non_holiday_sales, holiday_sales], labels=['Non-Holiday', 'Holiday'])
    axes[1, 0].set_title('Sales: Holiday vs Non-Holiday')
    axes[1, 0].set_ylabel('Weekly Sales ($)')
    
    # 5. Sales by Quarter
    quarterly_sales = df.groupby('Quarter')['Weekly_Sales'].mean()
    axes[1, 1].bar(quarterly_sales.index, quarterly_sales.values, color='orange', alpha=0.7)
    axes[1, 1].set_title('Average Sales by Quarter')
    axes[1, 1].set_xlabel('Quarter')
    axes[1, 1].set_ylabel('Average Weekly Sales ($)')
    axes[1, 1].set_xticks([1, 2, 3, 4])
    
    # 6. Sales by Month
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    axes[1, 2].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='purple')
    axes[1, 2].set_title('Average Sales by Month')
    axes[1, 2].set_xlabel('Month')
    axes[1, 2].set_ylabel('Average Weekly Sales ($)')
    axes[1, 2].set_xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig('walmart_sales_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_analysis_visualizations(df):
    """Create visualizations for feature analysis and relationships"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Analysis and Relationships', fontsize=16, fontweight='bold')
    
    # 1. Temperature vs Sales Scatter
    axes[0, 0].scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5, color='red')
    axes[0, 0].set_title('Temperature vs Weekly Sales')
    axes[0, 0].set_xlabel('Temperature (°F)')
    axes[0, 0].set_ylabel('Weekly Sales ($)')
    
    # Add trend line
    z = np.polyfit(df['Temperature'], df['Weekly_Sales'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['Temperature'], p(df['Temperature']), "r--", alpha=0.8)
    
    # 2. Fuel Price vs Sales
    axes[0, 1].scatter(df['Fuel_Price'], df['Weekly_Sales'], alpha=0.5, color='blue')
    axes[0, 1].set_title('Fuel Price vs Weekly Sales')
    axes[0, 1].set_xlabel('Fuel Price ($)')
    axes[0, 1].set_ylabel('Weekly Sales ($)')
    
    # 3. CPI vs Sales
    axes[0, 2].scatter(df['CPI'], df['Weekly_Sales'], alpha=0.5, color='green')
    axes[0, 2].set_title('CPI vs Weekly Sales')
    axes[0, 2].set_xlabel('Consumer Price Index')
    axes[0, 2].set_ylabel('Weekly Sales ($)')
    
    # 4. Unemployment vs Sales
    axes[1, 0].scatter(df['Unemployment'], df['Weekly_Sales'], alpha=0.5, color='orange')
    axes[1, 0].set_title('Unemployment vs Weekly Sales')
    axes[1, 0].set_xlabel('Unemployment Rate (%)')
    axes[1, 0].set_ylabel('Weekly Sales ($)')
    
    # 5. Temperature Distribution
    axes[1, 1].hist(df['Temperature'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].set_title('Temperature Distribution')
    axes[1, 1].set_xlabel('Temperature (°F)')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Sales by Temperature Category
    temp_sales = df.groupby('Temp_Category')['Weekly_Sales'].mean()
    axes[1, 2].bar(temp_sales.index, temp_sales.values, color='lightblue', alpha=0.7)
    axes[1, 2].set_title('Average Sales by Temperature Category')
    axes[1, 2].set_xlabel('Temperature Category')
    axes[1, 2].set_ylabel('Average Weekly Sales ($)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('walmart_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_and_interaction_visualizations(df):
    """Create correlation matrices and interaction visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Correlation Analysis and Feature Interactions', fontsize=16, fontweight='bold')
    
    # 1. Correlation Matrix Heatmap
    numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0, 0])
    axes[0, 0].set_title('Feature Correlation Matrix')
    
    # 2. Spearman Correlation Heatmap
    spearman_corr = df[numeric_cols].corr(method='spearman')
    sns.heatmap(spearman_corr, annot=True, cmap='viridis', center=0, 
                square=True, ax=axes[0, 1])
    axes[0, 1].set_title('Spearman Correlation Matrix')
    
    # 3. Temperature × Holiday Interaction
    temp_holiday_data = df.groupby(['Temp_Category', 'Holiday_Flag'])['Weekly_Sales'].mean().unstack()
    temp_holiday_data.plot(kind='bar', ax=axes[1, 0], color=['lightblue', 'orange'])
    axes[1, 0].set_title('Sales: Temperature × Holiday Interaction')
    axes[1, 0].set_xlabel('Temperature Category')
    axes[1, 0].set_ylabel('Average Weekly Sales ($)')
    axes[1, 0].legend(['Non-Holiday', 'Holiday'])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Store Performance by Quarter
    store_quarter_data = df.groupby(['Store', 'Quarter'])['Weekly_Sales'].mean().unstack()
    store_quarter_data.head(15).plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Store Performance by Quarter (Top 15 Stores)')
    axes[1, 1].set_xlabel('Store Number')
    axes[1, 1].set_ylabel('Average Weekly Sales ($)')
    axes[1, 1].legend(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('walmart_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_time_series_visualizations(df):
    """Create time series specific visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Time Series Analysis and Seasonality', fontsize=16, fontweight='bold')
    
    # 1. Sales Trend Over Time
    weekly_avg_sales = df.groupby('Date')['Weekly_Sales'].mean()
    axes[0, 0].plot(weekly_avg_sales.index, weekly_avg_sales.values, linewidth=2, color='blue')
    axes[0, 0].set_title('Weekly Sales Trend (Store Average)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Average Weekly Sales ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add trend line
    x_numeric = np.arange(len(weekly_avg_sales))
    z = np.polyfit(x_numeric, weekly_avg_sales.values, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(weekly_avg_sales.index, p(x_numeric), "r--", alpha=0.8, linewidth=2)
    
    # 2. Seasonal Decomposition (Monthly)
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    axes[0, 1].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='green')
    axes[0, 1].set_title('Monthly Seasonality Pattern')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Weekly Sales ($)')
    axes[0, 1].set_xticks(range(1, 13))
    
    # 3. Year-over-Year Comparison
    yearly_monthly = df.groupby(['Year', 'Month'])['Weekly_Sales'].mean().unstack()
    yearly_monthly.plot(kind='line', marker='o', ax=axes[1, 0], linewidth=2)
    axes[1, 0].set_title('Year-over-Year Monthly Comparison')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Weekly Sales ($)')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].legend(['2010', '2011', '2012'])
    
    # 4. Holiday Effect Over Time
    holiday_effect = df.groupby('Date').apply(lambda x: 
        x[x['Holiday_Flag'] == 1]['Weekly_Sales'].mean() - 
        x[x['Holiday_Flag'] == 0]['Weekly_Sales'].mean() if len(x[x['Holiday_Flag'] == 1]) > 0 else 0)
    
    axes[1, 1].plot(holiday_effect.index, holiday_effect.values, linewidth=2, color='red', marker='o')
    axes[1, 1].set_title('Holiday Effect Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Holiday Premium ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('walmart_time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_store_performance_visualizations(df):
    """Create store-specific performance visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Store Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Store Performance Ranking
    store_performance = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    axes[0, 0].bar(range(len(store_performance)), store_performance['mean'], 
                    yerr=store_performance['std'], capsize=5, color='lightgreen', alpha=0.7)
    axes[0, 0].set_title('Store Performance Ranking (Mean ± Std)')
    axes[0, 0].set_xlabel('Store Rank')
    axes[0, 0].set_ylabel('Average Weekly Sales ($)')
    
    # 2. Store Performance Distribution
    axes[0, 1].hist(store_performance['mean'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 1].set_title('Distribution of Store Performance')
    axes[0, 1].set_xlabel('Average Weekly Sales ($)')
    axes[0, 1].set_ylabel('Number of Stores')
    
    # 3. Store Performance vs Consistency
    axes[1, 0].scatter(store_performance['mean'], store_performance['std'], alpha=0.7, color='purple')
    axes[1, 0].set_title('Store Performance vs Consistency')
    axes[1, 0].set_xlabel('Average Weekly Sales ($)')
    axes[1, 0].set_ylabel('Sales Standard Deviation ($)')
    
    # Add store labels for top performers
    top_stores = store_performance.head(5)
    for idx, (store, row) in enumerate(top_stores.iterrows()):
        axes[1, 0].annotate(f'Store {store}', (row['mean'], row['std']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Store Performance by Quarter
    store_quarter = df.groupby(['Store', 'Quarter'])['Weekly_Sales'].mean().unstack()
    store_quarter.head(10).plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Store Performance by Quarter (Top 10)')
    axes[1, 1].set_xlabel('Store Number')
    axes[1, 1].set_ylabel('Average Weekly Sales ($)')
    axes[1, 1].legend(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('walmart_store_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_outlier_and_quality_visualizations(df):
    """Create outlier detection and data quality visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Outlier Detection and Data Quality Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sales Outliers Detection
    Q1 = df['Weekly_Sales'].quantile(0.25)
    Q3 = df['Weekly_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    axes[0, 0].scatter(df['Weekly_Sales'], df['Temperature'], alpha=0.5, color='blue')
    axes[0, 0].axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7, label='Lower Bound')
    axes[0, 0].axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7, label='Upper Bound')
    axes[0, 0].set_title('Sales Outliers Detection (IQR Method)')
    axes[0, 0].set_xlabel('Weekly Sales ($)')
    axes[0, 0].set_ylabel('Temperature (°F)')
    axes[0, 0].legend()
    
    # 2. Temperature Outliers
    temp_Q1 = df['Temperature'].quantile(0.25)
    temp_Q3 = df['Temperature'].quantile(0.75)
    temp_IQR = temp_Q3 - temp_Q1
    temp_lower = temp_Q1 - 1.5 * temp_IQR
    temp_upper = temp_Q3 + 1.5 * temp_IQR
    
    axes[0, 1].hist(df['Temperature'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(x=temp_lower, color='red', linestyle='--', alpha=0.7, label='Lower Bound')
    axes[0, 1].axvline(x=temp_upper, color='red', linestyle='--', alpha=0.7, label='Upper Bound')
    axes[0, 1].set_title('Temperature Distribution with Outlier Bounds')
    axes[0, 1].set_xlabel('Temperature (°F)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Missing Data Pattern
    missing_data = df.isnull().sum()
    axes[1, 0].bar(missing_data.index, missing_data.values, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Missing Data by Column')
    axes[1, 0].set_xlabel('Columns')
    axes[1, 0].set_ylabel('Missing Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Data Completeness Heatmap
    # Create a sample of the data for better visualization
    sample_df = df.sample(n=100, random_state=42)
    missing_matrix = sample_df.isnull().astype(int)
    
    sns.heatmap(missing_matrix.T, cbar=True, ax=axes[1, 1], cmap='YlOrRd')
    axes[1, 1].set_title('Data Completeness Sample (100 rows)')
    axes[1, 1].set_xlabel('Row Index')
    axes[1, 1].set_ylabel('Columns')
    
    plt.tight_layout()
    plt.savefig('walmart_data_quality.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Walmart Sales Forecasting - Comprehensive Data Dashboard', fontsize=18, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Sales Distribution (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Weekly Sales Distribution')
    ax1.set_xlabel('Weekly Sales ($)')
    ax1.set_ylabel('Frequency')
    
    # 2. Sales by Store (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    store_means = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    ax2.bar(range(len(store_means)), store_means.values, color='lightgreen', alpha=0.7)
    ax2.set_title('Average Sales by Store')
    ax2.set_xlabel('Store Number (Ranked)')
    ax2.set_ylabel('Average Weekly Sales ($)')
    
    # 3. Time Series Trend (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    weekly_avg_sales = df.groupby('Date')['Weekly_Sales'].mean()
    ax3.plot(weekly_avg_sales.index, weekly_avg_sales.values, linewidth=2, color='blue')
    ax3.set_title('Weekly Sales Trend')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Average Weekly Sales ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Correlation Matrix (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=ax4)
    ax4.set_title('Feature Correlations')
    
    # 5. Monthly Seasonality (bottom left)
    ax5 = fig.add_subplot(gs[2, :2])
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    ax5.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='orange')
    ax5.set_title('Monthly Seasonality')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Average Weekly Sales ($)')
    ax5.set_xticks(range(1, 13))
    
    # 6. Holiday Effect (bottom right)
    ax6 = fig.add_subplot(gs[2, 2:])
    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']
    ax6.boxplot([non_holiday_sales, holiday_sales], labels=['Non-Holiday', 'Holiday'])
    ax6.set_title('Holiday Effect on Sales')
    ax6.set_ylabel('Weekly Sales ($)')
    
    # 7. Temperature vs Sales (bottom)
    ax7 = fig.add_subplot(gs[3, :])
    ax7.scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5, color='red')
    ax7.set_title('Temperature vs Weekly Sales')
    ax7.set_xlabel('Temperature (°F)')
    ax7.set_ylabel('Weekly Sales ($)')
    
    # Add trend line
    z = np.polyfit(df['Temperature'], df['Weekly_Sales'], 1)
    p = np.poly1d(z)
    ax7.plot(df['Temperature'], p(df['Temperature']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('walmart_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all visualizations"""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("Creating overview visualizations...")
    create_overview_visualizations(df)
    
    print("Creating sales analysis visualizations...")
    create_target_analysis_visualizations(df)
    
    print("Creating feature analysis visualizations...")
    create_feature_analysis_visualizations(df)
    
    print("Creating correlation and interaction visualizations...")
    create_correlation_and_interaction_visualizations(df)
    
    print("Creating time series visualizations...")
    create_time_series_visualizations(df)
    
    print("Creating store performance visualizations...")
    create_store_performance_visualizations(df)
    
    print("Creating data quality visualizations...")
    create_outlier_and_quality_visualizations(df)
    
    print("Creating comprehensive dashboard...")
    create_summary_dashboard(df)
    
    print("All visualizations completed and saved!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Records: {len(df):,}")
    print(f"Total Stores: {df['Store'].nunique()}")
    print(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Weeks: {df['Date'].nunique()}")
    print(f"Average Sales: ${df['Weekly_Sales'].mean():,.0f}")
    print(f"Sales Range: ${df['Weekly_Sales'].min():,.0f} - ${df['Weekly_Sales'].max():,.0f}")
    print(f"Holiday Weeks: {df['Holiday_Flag'].sum():,} ({df['Holiday_Flag'].mean()*100:.1f}%)")
    print(f"Temperature Range: {df['Temperature'].min():.1f}°F - {df['Temperature'].max():.1f}°F")
    print("="*60)

if __name__ == "__main__":
    main()
