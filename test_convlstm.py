import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError

# Load preprocessed data and scaler
X_test = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/X_test.npy')
y_test = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/y_test.npy')
with open('C:/Users/91905/PycharmProjects/WeatherNWP/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load trained model with custom objects
model = load_model('C:/Users/91905/PycharmProjects/WeatherNWP/convlstm_weather_model.h5',
                   custom_objects={'MeanSquaredError': MeanSquaredError})

# Make predictions
y_pred = model.predict(X_test)  # Shape: (1747, 17, 25, 1)

# Inverse scale predictions and actual values (for temperature only)
y_test_reshaped = y_test.reshape(-1, 1)  # Flatten to (1747*17*25, 1)
y_pred_reshaped = y_pred.squeeze(-1).reshape(-1, 1)  # Squeeze (1747, 17, 25, 1) to (1747, 17, 25) then flatten
dummy = np.zeros((y_test_reshaped.shape[0], 1))  # Dummy pressure column
y_test_unscaled = scaler.inverse_transform(np.hstack([y_test_reshaped, dummy]))[:, 0]
y_pred_unscaled = scaler.inverse_transform(np.hstack([y_pred_reshaped, dummy]))[:, 0]

# Reshape back to spatial grid
y_test_unscaled = y_test_unscaled.reshape(y_test.shape)  # (1747, 17, 25)
y_pred_unscaled = y_pred_unscaled.reshape(y_test.shape)  # (1747, 17, 25)

# Calculate MSE on unscaled data
mse_unscaled = np.mean((y_test_unscaled - y_pred_unscaled) ** 2)
print(f"Unscaled Test MSE: {mse_unscaled}")

# Visualize: Compare actual vs predicted for first test sample
sample_idx = 0
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(y_test_unscaled[sample_idx], cmap='coolwarm')
plt.title('Actual Temperature (°C)')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(y_pred_unscaled[sample_idx], cmap='coolwarm')
plt.title('Predicted Temperature (°C)')
plt.colorbar()
plt.tight_layout()
plt.savefig('C:/Users/91905/PycharmProjects/WeatherNWP/prediction_plot.png')
plt.show()