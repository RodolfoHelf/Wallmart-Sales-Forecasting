# 🏪 Walmart Sales Forecasting Project

A comprehensive machine learning project for forecasting Walmart sales using advanced data processing, feature engineering, and multiple ML models.

## 📁 **Clean & Organized Project Structure**

```
Wallmart-Sales-Forecasting/
├── 📁 app/                    # FastAPI web application
│   ├── main.py               # Main API endpoints
│   ├── config.py             # Configuration settings
│   ├── database.py           # Database connections
│   └── models/               # Database models & schemas
│
├── 📁 data/                  # Data processing modules
│   ├── data_processor.py     # CSV loading & cleaning
│   ├── feature_engineering.py # Feature creation (111+ features)
│   └── Walmart.csv           # Raw input data
│
├── 📁 models/                # ML model training
│   ├── train_models.py       # Full MLflow training
│   └── quick_test.py         # Fast validation testing
│
├── 📁 scripts/               # Execution scripts
│   ├── 📁 pipeline/          # Full pipeline execution
│   │   ├── run_full_pipeline.py
│   │   ├── run_pipeline.bat
│   │   ├── run_pipeline.ps1
│   │   └── pipeline_requirements.txt
│   │
│   └── 📁 quick_test/        # Quick model testing
│       ├── run_quick_test.bat
│       └── run_quick_test.ps1
│
├── 📁 docs/                  # Documentation
│   ├── README.md             # This file
│   ├── STARTUP_GUIDE.md      # Setup & deployment guide
│   ├── 📁 pipeline/          # Pipeline documentation
│   │   └── PIPELINE_README.md
│   └── 📁 quick_test/        # Quick test documentation
│       └── QUICK_TEST_README.md
│
├── 📁 config/                # Configuration files
│   ├── docker-compose.yml    # Docker services
│   ├── env.example           # Environment variables template
│   ├── init.sql              # Database initialization
│   ├── requirements.txt      # Main dependencies
│   └── setup.py              # Package setup
│
├── 📁 outputs/               # Generated outputs
│   ├── pipeline_outputs/     # Full pipeline results
│   │   ├── processed_data.csv
│   │   ├── featured_data.csv
│   │   ├── trained_models/
│   │   └── pipeline_summary.json
│   └── pipeline.log          # Execution logs
│
├── 📁 eda/                   # Exploratory Data Analysis
│   └── [EDA files...]
│
├── 📁 mlflow/                # MLflow experiment tracking
├── 📁 tests/                 # Test files
└── run.py                    # Main application runner
```

## 🚀 **Quick Start Guide**

### **1. Full Pipeline Execution** (Recommended)
```bash
# Navigate to pipeline scripts
cd scripts/pipeline

# Run the complete pipeline
python run_full_pipeline.py
# OR
run_pipeline.bat          # Windows
# OR
.\run_pipeline.ps1        # PowerShell
```

### **2. Quick Model Testing**
```bash
# Navigate to quick test scripts
cd scripts/quick_test

# Run quick validation
python ../models/quick_test.py
# OR
run_quick_test.bat        # Windows
# OR
.\run_quick_test.ps1      # PowerShell
```

### **3. Start Web Dashboard**
```bash
# From project root
python run.py
```

## 📊 **What the Pipeline Does**

1. **📁 Data Processing**: Loads `Walmart.csv` → Cleans & validates → Saves to `outputs/processed_data.csv`
2. **🔧 Feature Engineering**: Creates 111+ features → Saves to `outputs/featured_data.csv`
3. **🤖 Model Training**: Trains 4 model types for 5 stores → Saves to `outputs/trained_models/`

## 🎯 **Key Features**

- ✅ **Clean Organization**: Logical folder structure
- ✅ **Fast Execution**: Pipeline completes in ~1 second
- ✅ **Rich Features**: 111+ engineered features
- ✅ **Multiple Models**: Trend, Seasonal, Moving Average, Feature-based
- ✅ **Easy Execution**: Batch files & PowerShell scripts
- ✅ **Comprehensive Outputs**: CSV files, JSON summaries, logs

## 📚 **Documentation**

- **📖 Main Guide**: `docs/README.md` (this file)
- **🚀 Pipeline Guide**: `docs/pipeline/PIPELINE_README.md`
- **⚡ Quick Test Guide**: `docs/quick_test/QUICK_TEST_README.md`
- **⚙️ Setup Guide**: `docs/STARTUP_GUIDE.md`

## 🔧 **Requirements**

```bash
# Install main dependencies
pip install -r config/requirements.txt

# Install pipeline dependencies
pip install -r scripts/pipeline/pipeline_requirements.txt
```

## 📁 **Output Files**

After running the pipeline, check `outputs/pipeline_outputs/`:
- `processed_data.csv` - Cleaned data (16 columns)
- `featured_data.csv` - Enhanced data (127 columns)
- `trained_models/` - Model results & artifacts
- `pipeline_summary.json` - Complete execution summary

## 🎉 **Benefits of New Structure**

- **🧹 Clean Root**: Only essential files in main directory
- **📁 Logical Grouping**: Scripts, docs, configs organized by purpose
- **🔍 Easy Navigation**: Find what you need quickly
- **📚 Clear Documentation**: Each component has its own docs folder
- **⚡ Fast Execution**: Scripts are in dedicated folders with clear paths

---

**🎯 Ready to use! Start with the pipeline scripts in `scripts/pipeline/` for the full experience, or use quick test in `scripts/quick_test/` for fast validation.**
