import netCDF4 as nc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load NetCDF
ds = nc.Dataset('C:/Users/91905/PycharmProjects/WeatherNWP/era5_gujarat_2023.nc')
temp = ds.variables['t2m'][:] - 273.15  # Kelvin to Celsius
press = ds.variables['msl'][:] / 100    # Pa to hPa
time = nc.num2date(ds.variables['valid_time'][:], ds.variables['valid_time'].units)

# Stack features into channels
data = np.stack([temp, press], axis=-1)  # Shape: (8760, 17, 25, 2)

# Normalize data
scaler = MinMaxScaler()
data_reshaped = data.reshape(-1, 2)  # Flatten spatial dims for scaling
data_scaled = scaler.fit_transform(data_reshaped).reshape(8760, 17, 25, 2)

# Create sequences (24 hours input -> 1 hour output)
X, y = [], []
for i in range(24, len(data_scaled) - 1):
    X.append(data_scaled[i-24:i])  # 24 timesteps, 17×25 grid, 2 channels
    y.append(data_scaled[i, :, :, 0])  # Next hour’s temperature grid
X, y = np.array(X), np.array(y)

# Split: Train (80%), Test (20%)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("X_train shape:", X_train.shape)  # e.g., (6988, 24, 17, 25, 2)
print("y_train shape:", y_train.shape)  # e.g., (6988, 17, 25)

# Save
np.save('C:/Users/91905/PycharmProjects/WeatherNWP/X_train.npy', X_train)
np.save('C:/Users/91905/PycharmProjects/WeatherNWP/X_test.npy', X_test)
np.save('C:/Users/91905/PycharmProjects/WeatherNWP/y_train.npy', y_train)
np.save('C:/Users/91905/PycharmProjects/WeatherNWP/y_test.npy', y_test)
with open('C:/Users/91905/PycharmProjects/WeatherNWP/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

ds.close()