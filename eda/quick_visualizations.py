# Walmart Sales Forecasting - Quick Visualizations
# Run this script to quickly see the key data patterns and insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def quick_insights():
    """Generate quick visualizations to show key data insights"""
    
    print("Loading Walmart sales data...")
    
    # Load data
    df = pd.read_csv('data/Walmart.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    print(f"Dataset loaded: {len(df):,} records, {df['Store'].nunique()} stores")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Walmart Sales Data - Key Insights at a Glance', fontsize=16, fontweight='bold')
    
    # 1. Sales Distribution
    axes[0, 0].hist(df['Weekly_Sales'], bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Weekly Sales Distribution')
    axes[0, 0].set_xlabel('Weekly Sales ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['Weekly_Sales'].mean(), color='red', linestyle='--', 
                        label=f'Mean: ${df["Weekly_Sales"].mean():,.0f}')
    axes[0, 0].legend()
    
    # 2. Sales by Store (Top 15)
    store_means = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    top_15_stores = store_means.head(15)
    axes[0, 1].bar(range(len(top_15_stores)), top_15_stores.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Top 15 Stores by Average Sales')
    axes[0, 1].set_xlabel('Store Rank')
    axes[0, 1].set_ylabel('Average Weekly Sales ($)')
    
    # 3. Sales Trend Over Time
    weekly_avg = df.groupby('Date')['Weekly_Sales'].mean()
    axes[0, 2].plot(weekly_avg.index, weekly_avg.values, linewidth=2, color='blue')
    axes[0, 2].set_title('Sales Trend Over Time (Store Average)')
    axes[0, 2].set_xlabel('Date')
    axes[0, 2].set_ylabel('Average Weekly Sales ($)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Holiday Effect
    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']
    
    axes[1, 0].boxplot([non_holiday_sales, holiday_sales], labels=['Non-Holiday', 'Holiday'])
    axes[1, 0].set_title('Holiday Effect on Sales')
    axes[1, 0].set_ylabel('Weekly Sales ($)')
    
    # Add mean values
    axes[1, 0].text(0.5, non_holiday_sales.mean(), f'${non_holiday_sales.mean():,.0f}', 
                     ha='center', va='bottom', fontweight='bold')
    axes[1, 0].text(1.5, holiday_sales.mean(), f'${holiday_sales.mean():,.0f}', 
                     ha='center', va='bottom', fontweight='bold')
    
    # 5. Monthly Seasonality
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='orange')
    axes[1, 1].set_title('Monthly Seasonality Pattern')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Weekly Sales ($)')
    axes[1, 1].set_xticks(range(1, 13))
    
    # 6. Temperature vs Sales
    axes[1, 2].scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5, color='red')
    axes[1, 2].set_title('Temperature vs Weekly Sales')
    axes[1, 2].set_xlabel('Temperature (°F)')
    axes[1, 2].set_ylabel('Weekly Sales ($)')
    
    # Add trend line
    z = np.polyfit(df['Temperature'], df['Weekly_Sales'], 1)
    p = np.poly1d(z)
    axes[1, 2].plot(df['Temperature'], p(df['Temperature']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('walmart_quick_insights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS FROM THE DATA")
    print("="*60)
    
    # Sales insights
    print(f"📊 SALES INSIGHTS:")
    print(f"   • Average weekly sales: ${df['Weekly_Sales'].mean():,.0f}")
    print(f"   • Sales range: ${df['Weekly_Sales'].min():,.0f} - ${df['Weekly_Sales'].max():,.0f}")
    print(f"   • Sales variation: {df['Weekly_Sales'].std() / df['Weekly_Sales'].mean() * 100:.1f}%")
    
    # Store insights
    print(f"\n🏪 STORE INSIGHTS:")
    print(f"   • Total stores: {df['Store'].nunique()}")
    print(f"   • Store performance range: {store_means.min():,.0f} - {store_means.max():,.0f}")
    print(f"   • Store variation: {store_means.max() / store_means.min():.1f}x")
    
    # Time insights
    print(f"\n⏰ TIME INSIGHTS:")
    print(f"   • Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   • Total weeks: {df['Date'].nunique()}")
    print(f"   • Holiday weeks: {df['Holiday_Flag'].sum():,} ({df['Holiday_Flag'].mean()*100:.1f}%)")
    
    # Feature insights
    print(f"\n🌡️ FEATURE INSIGHTS:")
    print(f"   • Temperature range: {df['Temperature'].min():.1f}°F - {df['Temperature'].max():.1f}°F")
    print(f"   • Fuel price range: ${df['Fuel_Price'].min():.2f} - ${df['Fuel_Price'].max():.2f}")
    print(f"   • CPI range: {df['CPI'].min():.1f} - {df['CPI'].max():.1f}")
    
    # Correlation insights
    print(f"\n🔗 CORRELATION INSIGHTS:")
    temp_corr = df['Temperature'].corr(df['Weekly_Sales'])
    fuel_corr = df['Fuel_Price'].corr(df['Weekly_Sales'])
    cpi_corr = df['CPI'].corr(df['Weekly_Sales'])
    holiday_corr = df['Holiday_Flag'].corr(df['Weekly_Sales'])
    
    print(f"   • Temperature vs Sales: {temp_corr:.3f}")
    print(f"   • Fuel Price vs Sales: {fuel_corr:.3f}")
    print(f"   • CPI vs Sales: {cpi_corr:.3f}")
    print(f"   • Holiday vs Sales: {holiday_corr:.3f}")
    
    # Business insights
    print(f"\n💼 BUSINESS INSIGHTS:")
    holiday_premium = (holiday_sales.mean() - non_holiday_sales.mean()) / non_holiday_sales.mean() * 100
    print(f"   • Holiday sales premium: +{holiday_premium:.1f}%")
    print(f"   • Best performing month: {monthly_sales.idxmax()} ({monthly_sales.max():,.0f})")
    print(f"   • Worst performing month: {monthly_sales.idxmin()} ({monthly_sales.min():,.0f})")
    
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = quick_insights()
    print(f"\n✅ Quick visualizations completed! Check 'walmart_quick_insights.png' for the chart.")
