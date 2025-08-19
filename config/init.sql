-- Initialize Walmart Forecasting Database

-- Create tables for sales data
CREATE TABLE IF NOT EXISTS stores (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100),
    store_type VARCHAR(50),
    store_size_sqft INTEGER,
    location_state VARCHAR(50),
    location_city VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name VARCHAR(100),
    dept_category VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS sales_data (
    id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    dept_id INTEGER REFERENCES departments(dept_id),
    date DATE NOT NULL,
    weekly_sales DECIMAL(12,2),
    is_holiday BOOLEAN DEFAULT FALSE,
    temperature DECIMAL(5,2),
    fuel_price DECIMAL(5,2),
    cpi DECIMAL(8,4),
    unemployment_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    dept_id INTEGER REFERENCES departments(dept_id),
    forecast_date DATE NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    predicted_sales DECIMAL(12,2),
    confidence_lower DECIMAL(12,2),
    confidence_upper DECIMAL(12,2),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    mape_score DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    store_id INTEGER REFERENCES stores(store_id),
    dept_id INTEGER REFERENCES departments(dept_id),
    mape_score DECIMAL(8,4),
    wape_score DECIMAL(8,4),
    bias_score DECIMAL(8,4),
    validation_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_sales_data_store_dept_date ON sales_data(store_id, dept_id, date);
CREATE INDEX IF NOT EXISTS idx_sales_data_date ON sales_data(date);
CREATE INDEX IF NOT EXISTS idx_forecasts_store_dept_date ON forecasts(store_id, dept_id, forecast_date);
CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name, model_version);

-- Insert sample store data
INSERT INTO stores (store_id, store_name, store_type, store_size_sqft, location_state, location_city) VALUES
(1, 'Store 1', 'Supercenter', 150000, 'NC', 'Charlotte'),
(2, 'Store 2', 'Supercenter', 140000, 'SC', 'Columbia'),
(3, 'Store 3', 'Neighborhood Market', 45000, 'NC', 'Raleigh'),
(4, 'Store 4', 'Supercenter', 160000, 'VA', 'Richmond'),
(5, 'Store 5', 'Supercenter', 155000, 'GA', 'Atlanta')
ON CONFLICT (store_id) DO NOTHING;

-- Insert sample department data
INSERT INTO departments (dept_id, dept_name, dept_category) VALUES
(1, 'Grocery', 'Food & Beverage'),
(2, 'Household', 'Home & Garden'),
(3, 'Electronics', 'Technology'),
(4, 'Clothing', 'Apparel'),
(5, 'Automotive', 'Automotive'),
(6, 'Pharmacy', 'Health & Wellness'),
(7, 'Garden', 'Home & Garden'),
(8, 'Toys', 'Entertainment'),
(9, 'Sports', 'Outdoor & Recreation'),
(10, 'Books', 'Entertainment')
ON CONFLICT (dept_id) DO NOTHING;

