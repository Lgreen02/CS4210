import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import KFold
# Load data
data = pd.read_csv('2012weather.csv')
pred_data = pd.read_csv('2023weather.csv')

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train_data, test_size=0.2, random_state=42)
# Separate features (X) and target (y)
train_y = train.iloc[:, 0]  # First column as target
train_X = train.drop(data.columns[0], axis=1)  # All other columns as features
val_y = val.iloc[:, 0]  # First column as target
val_X = val.drop(data.columns[0], axis=1)  # All other columns as features
test_y = test_data.iloc[:, 0]
test_X = test_data.drop(data.columns[0], axis=1)
pred_y = pred_data.iloc[:, 0]
pred_X = pred_data.drop(pred_data.columns[0], axis=1)

# Ensure the date column is in datetime format
train_X['Date'] = pd.to_datetime(train_X.iloc[:, 0])  # Assuming the date is the first column
val_X['Date'] = pd.to_datetime(val_X.iloc[:, 0])  # Assuming the date is the first column
test_X['Date'] = pd.to_datetime(test_X.iloc[:, 0])
pred_X['Date'] = pd.to_datetime(pred_X.iloc[:,0])

# Extract date components
for df in [train_X, test_X, val_X, pred_X]:
    df['Year'] = df['Date'].dt.year
    #df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    #df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop the original date column
train_X = train_X.drop(columns=['Date'])
test_X = test_X.drop(columns=['Date'])
val_X = val_X.drop(columns=['Date'])
train_X = train_X.drop(columns=['time'])
test_time = test_X['time']
test_X = test_X.drop(columns=['time'])
val_X = val_X.drop(columns=['time'])
pred_X = pred_X.drop(columns=['time'])
pred_X = pred_X.drop(columns=['Date'])

#future_pred_X = []
#for i in range(366):
    #future_pred_X.append([i])

# Scale the features for better neural network performance
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
print(val_X['DayOfYear'])
val_X_scaled = scaler.fit_transform(val_X)
test_X_scaled = scaler.transform(test_X)
pred_X_scaled = scaler.transform(pred_X)
# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
mse_scores = []
mae_scores = []

for train_index, val_index in kf.split(train_X_scaled):
    print(f"Training Fold {fold}...")

    # Split data into train and validation sets for this fold
    X_train, X_val = train_X_scaled[train_index], train_X_scaled[val_index]
    y_train, y_val = train_y.iloc[train_index], train_y.iloc[val_index]

    # Build the neural network model
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),  # First hidden layer
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),  # Second hidden layer
        Dense(1, activation='linear')  # Output layer for regression
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Use a lower number of epochs for K-Fold to reduce computation
        batch_size=32,
        verbose=1
    )

    # Predict on validation set
    y_val_pred = model.predict(X_val)
    
    # Evaluate the model
    mse_fold = mean_squared_error(y_val, y_val_pred)
    mae_fold = mean_absolute_error(y_val, y_val_pred)

    mse_scores.append(mse_fold)
    mae_scores.append(mae_fold)

    print(f"Fold {fold} - MSE: {mse_fold}, MAE: {mae_fold}")
    fold += 1

# Compute the average performance across folds
avg_mse = np.mean(mse_scores)
avg_mae = np.mean(mae_scores)

print(f"Average MSE across folds: {avg_mse}")
print(f"Average MAE across folds: {avg_mae}")

# Final Model
print("Training final model on entire training dataset...")
model.fit(
    train_X_scaled, train_y,
    validation_data=(val_X_scaled, val_y),
    epochs=200,
    batch_size=32,
    verbose=1
)

# Predict on test set
y_pred_nn = model.predict(test_X_scaled)
accurate_results = 0
errors = []

y_pred_future = model.predict(pred_X_scaled)
# Ensure test_y is aligned with the predicted values
#data_one_week_y = data_one_week_y.reset_index(drop=True)
#y_pred_future = pd.Series(y_pred_future)  # Convert predictions to Series for consistency

# Extract time values from the original test dataset
time_axis = pred_X['DayOfYear'] # Assuming the first column contains date/time

# Plot actual vs predicted temperatures
plt.figure(figsize=(12, 6))
plt.plot(time_axis, pred_y, label='Actual Temperature', color='blue', alpha=0.7)
plt.plot(time_axis, y_pred_future, label='Predicted Temperature', color='red', linestyle='--', alpha=0.7)

# Add plot titles and labels
plt.title('Temperature Prediction vs Actual', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Temperature (°F)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Display the plot
plt.show()

for n, pred in zip(test_y, y_pred_nn):
    print("Actual: ", n, " | | Predicted: ", pred)
    if abs(n - pred) < 8:
        accurate_results += 1
    errors.append(abs(n-pred))
print("The % of results within 2 degrees: ", (accurate_results / len(test_y)))
# Evaluate final performance
final_mse = mean_squared_error(test_y, y_pred_nn)
final_mae = mean_absolute_error(test_y, y_pred_nn)
print("Mean absolute error: ",final_mae)
"""
# Create the box plot
plt.boxplot(errors)
plt.title("Box and Whisker Plot")
plt.show()
print(f"Final Model - Test MSE: {final_mse}")
print(f"Final Model - Test MAE: {final_mae}")
"""