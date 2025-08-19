# 🚀 Walmart Sales Forecasting - Full Pipeline

## Overview
The Full Pipeline script automates the complete Walmart sales forecasting workflow from raw data to trained models. It processes the `Walmart.csv` file through three main stages and saves outputs at each step.

## 🔄 Pipeline Stages

### **Stage 1: Data Processing** 📊
- **Input**: `data/Walmart.csv`
- **Process**: Load, clean, validate, and prepare raw data
- **Output**: `pipeline_outputs/processed_data.csv`
- **Features**: Data cleaning, type conversion, missing value handling

### **Stage 2: Feature Engineering** 🔧
- **Input**: Processed data from Stage 1
- **Process**: Create temporal, lag, rolling, and interaction features
- **Output**: `pipeline_outputs/featured_data.csv`
- **Features**: 50+ engineered features for enhanced model performance

### **Stage 3: Model Training** 🤖
- **Input**: Featured data from Stage 2
- **Process**: Train multiple forecasting models for store-department combinations
- **Output**: `pipeline_outputs/trained_models/`
- **Models**: SARIMAX, LightGBM, XGBoost, Prophet, Ensemble

## 🚀 Quick Start

### **Method 1: Direct Python Execution**
```bash
python run_full_pipeline.py
```

### **Method 2: Windows Batch File**
```bash
run_pipeline.bat
```

### **Method 3: PowerShell Script**
```powershell
.\run_pipeline.ps1
```

## 📦 Prerequisites

### **Install Dependencies**
```bash
pip install -r pipeline_requirements.txt
```

### **Required Files**
- ✅ `data/Walmart.csv` - Input data file
- ✅ `data/data_processor.py` - Data processing module
- ✅ `data/feature_engineering.py` - Feature engineering module
- ✅ `models/train_models.py` - Model training module

## 📁 Output Structure

```
pipeline_outputs/
├── processed_data.csv          # Stage 1 output
├── featured_data.csv           # Stage 2 output
├── trained_models/             # Stage 3 outputs
│   ├── training_results.json   # Training summary
│   └── [model artifacts]      # Trained models
├── pipeline_summary.json       # Complete pipeline summary
└── pipeline.log               # Detailed execution log
```

## 🔍 Pipeline Outputs

### **1. Processed Data** (`processed_data.csv`)
- Cleaned and validated Walmart sales data
- Proper data types and formats
- Missing values handled
- Ready for feature engineering

### **2. Featured Data** (`featured_data.csv`)
- Original data + 50+ engineered features
- Temporal features (year, month, quarter, etc.)
- Lag features (1-week, 2-week, 4-week)
- Rolling statistics (mean, std, min, max)
- Holiday and seasonal indicators
- Interaction features

### **3. Trained Models** (`trained_models/`)
- Model training results for each store-department
- Performance metrics and validation scores
- Model artifacts and configurations
- Training logs and summaries

### **4. Pipeline Summary** (`pipeline_summary.json`)
- Complete execution summary
- Stage-by-stage performance metrics
- Data transformation statistics
- Model training results
- Execution timestamps and durations

## 📊 Expected Data Flow

```
Walmart.csv (Raw)
       ↓
[Data Processing]
       ↓
processed_data.csv
       ↓
[Feature Engineering]
       ↓
featured_data.csv
       ↓
[Model Training]
       ↓
Trained Models + Results
```

## ⚙️ Configuration Options

### **Input File Path**
```python
# In run_full_pipeline.py, modify the input path
pipeline = WalmartPipelineRunner("path/to/your/Walmart.csv")
```

### **Output Directory**
```python
# The pipeline automatically creates:
# - pipeline_outputs/
# - pipeline_outputs/trained_models/
```

### **Model Training Scope**
```python
# Currently trains models for first 5 store-department combinations
# Modify max_combinations in _run_model_training() for different scope
max_combinations = min(5, len(store_dept_combinations))
```

## 🔧 Customization

### **Adding New Features**
```python
# In feature_engineering.py, add new feature creation methods
def create_custom_features(self) -> pd.DataFrame:
    # Your custom feature logic here
    pass
```

### **Modifying Model Training**
```python
# In train_models.py, customize model parameters and training logic
def _train_custom_model(self, data, store_id, dept_id):
    # Your custom training logic here
    pass
```

### **Pipeline Stages**
```python
# Add new stages in run_full_pipeline.py
def _run_custom_stage(self, data):
    # Your custom stage logic here
    pass
```

## 📈 Performance Monitoring

### **Execution Times**
- **Data Processing**: Typically 5-15 seconds
- **Feature Engineering**: Typically 10-30 seconds
- **Model Training**: Typically 30-120 seconds (depends on data size)
- **Total Pipeline**: Typically 1-3 minutes

### **Memory Usage**
- **Processed Data**: ~50-100 MB
- **Featured Data**: ~100-200 MB
- **Model Training**: Varies by model complexity

### **Data Volume**
- **Input**: Walmart.csv (typically 100K-1M+ rows)
- **Output**: Enhanced dataset with 50+ features
- **Models**: One set per store-department combination

## 🚨 Troubleshooting

### **Common Issues**

#### 1. **Input File Not Found**
```bash
❌ Error: Input file not found: data/Walmart.csv
```
**Solution**: Ensure `Walmart.csv` is in the `data/` directory

#### 2. **Missing Dependencies**
```bash
❌ ModuleNotFoundError: No module named 'pandas'
```
**Solution**: Install requirements with `pip install -r pipeline_requirements.txt`

#### 3. **Memory Issues**
```bash
❌ MemoryError during feature engineering
```
**Solution**: Reduce data size or optimize feature creation

#### 4. **Model Training Failures**
```bash
❌ Model training failed for specific store-dept
```
**Solution**: Check data quality and model parameters

### **Debug Mode**
```python
# Add debug logging in run_full_pipeline.py
logging.basicConfig(level=logging.DEBUG)
```

### **Log Files**
- **`pipeline.log`**: Detailed execution logs
- **`pipeline_summary.json`**: Structured results summary

## 🔄 Integration with Other Tools

### **Before Running Pipeline**
```bash
# 1. Quick validation test
python models/quick_test.py

# 2. Check data quality
python data/data_processor.py

# 3. Run full pipeline
python run_full_pipeline.py
```

### **After Pipeline Completion**
```bash
# 1. View results
cat pipeline_outputs/pipeline_summary.json

# 2. Start FastAPI dashboard
python run.py

# 3. Access MLflow tracking
# Open http://localhost:5000 in browser
```

## 📚 API Integration

### **Pipeline Results API**
```python
# Access pipeline results programmatically
from run_full_pipeline import WalmartPipelineRunner

pipeline = WalmartPipelineRunner()
results = pipeline.run_full_pipeline()
status = pipeline.get_pipeline_status()
```

### **Individual Stage Execution**
```python
# Run individual stages
processed_data = pipeline._run_data_processing()
featured_data = pipeline._run_feature_engineering(processed_data)
model_results = pipeline._run_model_training(featured_data)
```

## 🎯 Use Cases

### **Development & Testing**
- Rapid iteration on feature engineering
- Model validation and comparison
- Data quality assessment

### **Production Deployment**
- Automated model retraining
- Batch processing workflows
- CI/CD pipeline integration

### **Research & Analysis**
- Feature importance analysis
- Model performance benchmarking
- Data transformation validation

## 🚀 Next Steps

After successful pipeline execution:

1. **Review Results**: Check `pipeline_summary.json` for insights
2. **Validate Models**: Test trained models on validation data
3. **Deploy Dashboard**: Start FastAPI application with `python run.py`
4. **Monitor Performance**: Use MLflow for experiment tracking
5. **Iterate**: Modify features and models based on results

## 📞 Support

### **Documentation**
- **Pipeline Logs**: Check `pipeline.log` for detailed execution info
- **Results Summary**: Review `pipeline_summary.json` for structured results
- **Code Comments**: Inline documentation in all pipeline scripts

### **Common Commands**
```bash
# Check pipeline status
python -c "from run_full_pipeline import WalmartPipelineRunner; print(WalmartPipelineRunner().get_pipeline_status())"

# View pipeline outputs
ls -la pipeline_outputs/

# Check pipeline logs
tail -f pipeline.log
```

---

**Happy Pipeline Execution! 🚀📊🔧**

For additional support, check the pipeline logs or create an issue in the project repository.
