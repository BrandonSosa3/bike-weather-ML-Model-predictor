Bike Share Demand Forecasting

A machine learning system for predicting hourly bike share demand based on weather conditions and temporal patterns.

Live Demo Streamlit App: https://bike-weather-ml-model-predictor-p4ziyczsz9mvg2ccetgmgv.streamlit.app

Try the application directly in your browser no installation required.

Overview

This project develops a Random Forest regression model to forecast bike share usage in urban environments. Using two years of historical data from Washington DC's Capital Bikeshare system, the model achieves 67% prediction accuracy (R² = 0.667) and provides actionable insights for operational optimization.

Key Results

- Model Performance: Random Forest achieves R² of 0.667, explaining 67% of variance in bike usage
- Prediction Error: RMSE of 127 bikes per hour, 48-bike improvement over linear baseline
- Feature Importance: Hour of day accounts for 48.6% of predictive power
- User Behavior: Weekday commuters are 5x more weather-resistant than weekend recreational riders

Dataset

- Source: Capital Bikeshare (Washington DC) via Kaggle
- Time Period: 2011-2012 (17,379 hourly records)
- Features: Weather conditions (temperature, humidity, wind speed), temporal variables (hour, day, month, season), and engineered features (weather comfort index, rush hour indicators)

Methodology

Data Pipeline
1. Data acquisition from Kaggle API
2. Quality validation and cleaning
3. Feature engineering (weather comfort indices, temporal indicators)
4. Database storage using SQLite and SQLAlchemy

Machine Learning Approach
1. Baseline Model: Linear Regression (R² = 0.367)
2. Advanced Model: Random Forest Regressor (R² = 0.667)
3. Validation: Time-aware train/test split (80/20)
4. Deployment: Interactive Streamlit web application

Key Findings

Temporal Dominance
Hour of day is the strongest predictor (48.6% feature importance), indicating that commuting patterns drive demand more than weather conditions.

Weather Impact Patterns
 Clear weather: 206 bikes/hour average
 Heavy rain: 62 bikes/hour average
 Temperature shows non-linear relationship with optimal usage around 20-25°C

User Segmentation
 Weekday commuters maintain 63% of normal usage during heavy rain
 Weekend recreational riders drop to 14% of normal usage during heavy rain
 Rush hour periods (7-9 AM, 5-7 PM) show 3x baseline usage

Business Recommendations

Operational Efficiency
 Focus bike rebalancing during rush hour windows
 Schedule maintenance during low-demand periods (5-7 AM)
 Implement weather-responsive staffing for weekend operations


Technical Stack
 Language: Python 3.10
 Data Processing: Pandas, NumPy
 Machine Learning: Scikit-learn (Random Forest)
 Database: SQLite, SQLAlchemy
 Visualization: Streamlit, Plotly
 Development: Git, Virtual Environments

Model Performance

Metric              Linear Regression     Random Forest
R² Score            0.367                 0.667
RMSE                175.2 bikes           127.2 bikes
MAE                 131.5 bikes           89.3 bikes

