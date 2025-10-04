
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# meteorological data
data = {
    'Humidity': [30, 40, 50, 60, 70, 65, 55, 45, 35, 75, 80, 85, 90, 95, 100],
    'WindSpeed': [5, 7, 6, 8, 7, 9, 5, 6, 4, 8, 7, 10, 9, 11, 10],  # km/h
    'Temperature': [20, 22, 24, 26, 28, 27, 25, 23, 21, 29, 30, 32, 31, 33, 34],
    'Precipitation': [0, 1, 0, 2, 1, 3, 1, 0, 0, 4, 3, 5, 4, 6, 7]  # mm
}
df = pd.DataFrame(data)

# Features and targets
X = df[['Humidity', 'WindSpeed']].values
y_temp = df['Temperature'].values
y_precip = df['Precipitation'].values

# Feature scaling (important for SVR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVR model for Temperature
svr_temp = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_temp.fit(X_scaled, y_temp)
pred_temp = svr_temp.predict(X_scaled)

# SVR model for Precipitation
svr_precip = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_precip.fit(X_scaled, y_precip)
pred_precip = svr_precip.predict(X_scaled)

# Calculate evaluation metrics for Temperature
mse_temp = mean_squared_error(y_temp, pred_temp)
rmse_temp = np.sqrt(mse_temp)
mae_temp = mean_absolute_error(y_temp, pred_temp)
r2_temp = r2_score(y_temp, pred_temp)

print("Temperature Prediction Metrics:")
print(f"Mean Squared Error (MSE): {mse_temp:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse_temp:.3f}")
print(f"Mean Absolute Error (MAE): {mae_temp:.3f}")
print(f"R-squared (R²): {r2_temp:.3f}\n")

# Calculate evaluation metrics for Precipitation
mse_precip = mean_squared_error(y_precip, pred_precip)
rmse_precip = np.sqrt(mse_precip)
mae_precip = mean_absolute_error(y_precip, pred_precip)
r2_precip = r2_score(y_precip, pred_precip)

print("Precipitation Prediction Metrics:")
print(f"Mean Squared Error (MSE): {mse_precip:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse_precip:.3f}")
print(f"Mean Absolute Error (MAE): {mae_precip:.3f}")
print(f"R-squared (R²): {r2_precip:.3f}\n")

# Plot Temperature actual vs predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['Humidity'], y_temp, color="blue", label="Actual Temp")
plt.scatter(df['Humidity'], pred_temp, color="red", label="Predicted Temp", marker='x')
plt.xlabel("Humidity")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Prediction (SVR)")
plt.legend()
plt.grid(True)

# Plot Precipitation actual vs predicted
plt.subplot(1, 2, 2)
plt.scatter(df['Humidity'], y_precip, color="green", label="Actual Precip")
plt.scatter(df['Humidity'], pred_precip, color="orange", label="Predicted Precip", marker='x')
plt.xlabel("Humidity")
plt.ylabel("Precipitation (mm)")
plt.title("Precipitation Prediction (SVR)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
