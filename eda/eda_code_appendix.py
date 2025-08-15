# Walmart Sales Forecasting - EDA Code Appendix
# This file contains the exact code snippets used for key tables and plots in the EDA analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
def load_data():
    """Load and perform initial data inspection"""
    df = pd.read_csv('data/Walmart.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.3f} MB")
    return df

# 1. Data Loading Summary
def data_loading_summary(df):
    """Generate data loading summary statistics"""
    
    # Basic info
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'duplicates': df.duplicated().sum(),
        'store_date_duplicates': df.duplicated(subset=['Store', 'Date']).sum()
    }
    
    # Store analysis
    store_counts = df['Store'].value_counts()
    summary['stores'] = len(store_counts)
    summary['weeks_per_store'] = store_counts.iloc[0]
    
    # Date range
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    summary['date_range'] = {
        'start': df['Date'].min().strftime('%Y-%m-%d'),
        'end': df['Date'].max().strftime('%Y-%m-%d'),
        'duration_weeks': (df['Date'].max() - df['Date'].min()).days // 7
    }
    
    return summary, df

# 2. Quality and Integrity Audit
def quality_audit(df):
    """Perform comprehensive data quality audit"""
    
    # Missingness analysis
    missing_summary = df.isnull().sum()
    missing_rate = missing_summary / len(df) * 100
    
    # Outlier detection using robust z-score (MAD)
    def robust_zscore(x):
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        return np.abs(x - median) / (1.4826 * mad)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        if col == 'Store':
            continue
        z_scores = robust_zscore(df[col])
        extreme_count = np.sum(z_scores > 3)
        outlier_summary[col] = {
            'extreme_rate': extreme_count / len(df),
            'top_offenders': df[z_scores > 3][col].nlargest(3).tolist()
        }
    
    # IQR analysis
    iqr_summary = {}
    for col in numeric_cols:
        if col == 'Store':
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_summary[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'outliers': len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
        }
    
    return missing_summary, missing_rate, outlier_summary, iqr_summary

# 3. Target Assessment
def target_assessment(df):
    """Analyze the target variable (Weekly_Sales)"""
    
    target = df['Weekly_Sales']
    
    # Basic statistics
    stats_summary = {
        'mean': target.mean(),
        'median': target.median(),
        'std': target.std(),
        'skewness': target.skew(),
        'range': [target.min(), target.max()]
    }
    
    # Quantiles
    quantiles = target.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    
    # Holiday vs Non-Holiday analysis
    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']
    
    holiday_effect = (holiday_sales.mean() - non_holiday_sales.mean()) / non_holiday_sales.mean()
    
    # Store variation
    store_means = df.groupby('Store')['Weekly_Sales'].mean()
    store_variation = store_means.max() / store_means.min()
    
    return stats_summary, quantiles, holiday_effect, store_variation

# 4. Numeric Profiling
def numeric_profiling(df):
    """Generate comprehensive numeric feature profiles"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Store']
    
    # Detailed statistics
    stats_df = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    
    # Additional metrics
    additional_metrics = {}
    for col in numeric_cols:
        additional_metrics[col] = {
            'distinct_count': df[col].nunique(),
            'zero_share': (df[col] == 0).mean(),
            'skewness': df[col].skew()
        }
    
    # Correlation analysis
    correlation_matrix = df[numeric_cols].corr()
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    # VIF calculation
    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    # Note: variance_inflation_factor requires statsmodels
    # from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    return stats_df, additional_metrics, correlation_matrix, spearman_corr

# 5. Categorical Profiling
def categorical_profiling(df):
    """Analyze categorical features"""
    
    # Holiday_Flag analysis
    holiday_dist = df['Holiday_Flag'].value_counts()
    holiday_rate = holiday_dist[1] / len(df)
    
    # Store analysis
    store_sales = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    top_stores = store_sales.head(10)
    
    # Target rate by category
    holiday_sales_avg = df[df['Holiday_Flag'] == 1]['Weekly_Sales'].mean()
    non_holiday_sales_avg = df[df['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
    
    return holiday_dist, holiday_rate, store_sales, top_stores, holiday_sales_avg, non_holiday_sales_avg

# 6. Time Series Analysis
def time_series_analysis(df):
    """Analyze temporal patterns and seasonality"""
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Monthly aggregation
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['Weekly_Sales'].agg(['mean', 'count'])
    
    # Quarterly analysis
    df['Quarter'] = df['Date'].dt.quarter
    quarterly_sales = df.groupby('Quarter')['Weekly_Sales'].mean()
    
    # Trend analysis
    df['Week_Number'] = (df['Date'] - df['Date'].min()).dt.days // 7
    trend_model = LinearRegression()
    trend_model.fit(df[['Week_Number']], df['Weekly_Sales'])
    weekly_trend = trend_model.coef_[0]
    
    return monthly_sales, quarterly_sales, weekly_trend

# 7. Interactions and Nonlinearity
def interaction_analysis(df):
    """Analyze feature interactions and nonlinear relationships"""
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['Store', 'Weekly_Sales']]
    
    # Correlation with target
    target_correlations = {}
    for col in numeric_cols:
        target_correlations[col] = df[col].corr(df['Weekly_Sales'])
    
    # Mutual information (requires target to be continuous)
    X = df[numeric_cols]
    y = df['Weekly_Sales']
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_summary = dict(zip(numeric_cols, mi_scores))
    
    # Two-way interactions (example: Temperature × Holiday)
    df['Temp_Holiday_Interaction'] = df['Temperature'] * df['Holiday_Flag']
    interaction_corr = df['Temp_Holiday_Interaction'].corr(df['Weekly_Sales'])
    
    return target_correlations, mi_summary, interaction_corr

# 8. Visualization Functions
def create_visualizations(df):
    """Create key visualizations for the EDA report"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Weekly Sales Distribution
    axes[0, 0].hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Weekly Sales Distribution')
    axes[0, 0].set_xlabel('Weekly Sales ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Sales by Store
    store_means = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(store_means)), store_means.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Average Sales by Store')
    axes[0, 1].set_xlabel('Store Number')
    axes[0, 1].set_ylabel('Average Weekly Sales ($)')
    
    # 3. Sales by Month
    df['Month'] = pd.to_datetime(df['Date'], format='%d-%m-%Y').dt.month
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    axes[1, 0].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='orange')
    axes[1, 0].set_title('Average Sales by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Weekly Sales ($)')
    axes[1, 0].set_xticks(range(1, 13))
    
    # 4. Temperature vs Sales Scatter
    axes[1, 1].scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5, color='red')
    axes[1, 1].set_title('Temperature vs Weekly Sales')
    axes[1, 1].set_xlabel('Temperature (°F)')
    axes[1, 1].set_ylabel('Weekly Sales ($)')
    
    plt.tight_layout()
    plt.savefig('walmart_eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()

# 9. Data Quality Score Calculation
def calculate_data_quality_score(df):
    """Calculate overall data quality score"""
    
    scores = {}
    
    # Missing data score
    missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
    scores['missing_data'] = 1 - missing_rate
    
    # Duplicate score
    duplicate_rate = df.duplicated().sum() / len(df)
    scores['duplicates'] = 1 - duplicate_rate
    
    # Data type consistency score
    type_consistency = 1.0  # Assuming all data types are appropriate
    
    # Outlier score (using IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_rates = []
    for col in numeric_cols:
        if col == 'Store':
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
        outlier_rate = outliers / len(df)
        outlier_rates.append(1 - outlier_rate)
    
    scores['outliers'] = np.mean(outlier_rates) if outlier_rates else 1.0
    
    # Overall score
    overall_score = np.mean(list(scores.values()))
    
    return scores, overall_score

# Main execution function
def run_complete_eda():
    """Run the complete EDA analysis"""
    
    print("Loading data...")
    df = load_data()
    
    print("Performing data loading summary...")
    summary, df = data_loading_summary(df)
    print(f"Data loading summary: {summary}")
    
    print("Performing quality audit...")
    missing_summary, missing_rate, outlier_summary, iqr_summary = quality_audit(df)
    print(f"Missing data summary:\n{missing_summary}")
    print(f"Outlier summary:\n{outlier_summary}")
    
    print("Assessing target variable...")
    stats_summary, quantiles, holiday_effect, store_variation = target_assessment(df)
    print(f"Target statistics: {stats_summary}")
    print(f"Holiday effect: {holiday_effect:.3f}")
    print(f"Store variation: {store_variation:.1f}x")
    
    print("Creating visualizations...")
    create_visualizations(df)
    
    print("Calculating data quality score...")
    quality_scores, overall_score = calculate_data_quality_score(df)
    print(f"Overall data quality score: {overall_score:.3f}")
    
    print("EDA analysis complete!")
    return df, summary, quality_scores

# Example usage
if __name__ == "__main__":
    # Run the complete EDA
    df, summary, quality_scores = run_complete_eda()
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS SUMMARY")
    print("="*50)
    print(f"Dataset: {summary['rows']} rows, {summary['columns']} columns")
    print(f"Stores: {summary['stores']} stores with {summary['weeks_per_store']} weeks each")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Data quality score: {quality_scores:.3f}")
    print("="*50)
