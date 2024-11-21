import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data, skipping the first two rows
data = pd.read_csv('historical_weather_data.csv', skiprows=2)

# Rename columns appropriately
data.columns = ['time', 'temperature_2m']

# Check column names
print(data.columns)
print()

# Inspect the first few rows of data
print(data.head())

# Preprocessing
data.fillna(method='ffill', inplace=True)  # Handle missing values

# Convert 'time' to datetime and set as index
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

# Sort the data by date to ensure monotonic order
data = data.sort_index()

# Infer frequency and explicitly set it
freq = pd.infer_freq(data.index)
if freq is None:
    freq = 'H'  # Assuming hourly frequency, adjust as necessary
data = data.asfreq(freq)

# Feature Engineering
data['Month'] = data.index.month

# Define features and target
X = data[['Month']]
y = data['temperature_2m']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print()
print(f'Linear Regression MAE: {mean_absolute_error(y_test, y_pred_lin)}')

# ARIMA Model
arima_model = ARIMA(y_train, order=(5,1,0))
arima_result = arima_model.fit()
y_pred_arima = arima_result.forecast(steps=len(y_test))
print()
print(f'ARIMA MAE: {mean_absolute_error(y_test, y_pred_arima)}')

# Ridge Regression Model
ridge = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train, y_train)
y_pred_ridge = grid_search.predict(X_test)
print()
print(f'Ridge Regression MAE: {mean_absolute_error(y_test, y_pred_ridge)}')