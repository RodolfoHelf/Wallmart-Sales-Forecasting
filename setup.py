"""
Setup script for Walmart Sales Forecasting Dashboard
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="walmart-sales-forecasting",
    version="1.0.0",
    author="Walmart Data Science Team",
    author_email="data-science@walmart.com",
    description="ML-powered sales forecasting dashboard for Walmart South Atlantic Division",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/walmart/walmart-sales-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "ml": [
            "mlflow>=2.0.0",
            "lightgbm>=4.0.0",
            "xgboost>=2.0.0",
            "prophet>=1.0.0",
            "statsmodels>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "walmart-forecast=app.main:main",
            "train-models=models.train_models:main",
            "process-data=data.data_processor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.sql"],
    },
)









