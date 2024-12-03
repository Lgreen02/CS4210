import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load data
data = pd.read_csv('historical_large_max.csv')
data_one_week = pd.read_csv('test_temp.csv')
# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Separate features (X) and target (y)
train_y = train_data.iloc[:, 0]  # First column as target
train_X = train_data.drop(data.columns[0], axis=1)  # All other columns as features
test_y = test_data.iloc[:, 0]
test_X = test_data.drop(data.columns[0], axis=1)
data_one_week_y = data_one_week.iloc[:, 0]
data_one_week_X = data_one_week.drop(data.columns[0], axis=1)
# Ensure the date column is in datetime format
train_X['Date'] = pd.to_datetime(train_X.iloc[:, 0])  # Assuming the date is the first column
test_X['Date'] = pd.to_datetime(test_X.iloc[:, 0])
data_one_week_X['Date'] = pd.to_datetime(data_one_week_X.iloc[:, 0])
# Extract date components
for df in [train_X, test_X,data_one_week_X]:
    #df['Year'] = df['Date'].dt.year
    #df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    #df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop the original date column
train_X = train_X.drop(columns=['Date'])
test_X = test_X.drop(columns=['Date'])
train_X = train_X.drop(columns=['time'])
test_time = data_one_week_X['time']
test_X = test_X.drop(columns=['time'])
data_one_week_X = data_one_week_X.drop(columns=['time'])
data_one_week_X = data_one_week_X.drop(columns=['Date'])
# Now train_X and test_X are ready with transformed date features
print(train_X.head())
print(test_X.head())
print(train_y.head())
lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)
y_pred_lin = lin_reg.predict(data_one_week_X)
print()
print(f'Linear Regression MAE: {mean_absolute_error(data_one_week_y, y_pred_lin)}')

# Ridge Regression Model
ridge = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(train_X, train_y)
y_pred_ridge = grid_search.predict(test_X)
print()
print(f'Ridge Regression MAE: {mean_absolute_error(test_y, y_pred_ridge)}')   
import matplotlib.pyplot as plt

# Ensure test_y is aligned with the predicted values
data_one_week_y = data_one_week_y.reset_index(drop=True)
y_pred_lin = pd.Series(y_pred_lin)  # Convert predictions to Series for consistency

# Extract time values from the original test dataset
time_axis = data_one_week_X['DayOfYear']  # Assuming the first column contains date/time

# Plot actual vs predicted temperatures
plt.figure(figsize=(12, 6))
plt.plot(time_axis, data_one_week_y, label='Actual Temperature', color='blue', alpha=0.7)
plt.plot(time_axis, y_pred_lin, label='Predicted Temperature', color='red', linestyle='--', alpha=0.7)

# Add plot titles and labels
plt.title('Temperature Prediction vs Actual', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Temperature (°F)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Display the plot
plt.show()

"""
# ARIMA Model
arima_model = ARIMA(y_train, order=(5,1,0))
arima_result = arima_model.fit()
y_pred_arima = arima_result.forecast(steps=len(y_test))
print()
print(f'ARIMA MAE: {mean_absolute_error(y_test, y_pred_arima)}')



#Support Vector Matrix
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)

print(regr.predict(X_test))
"""