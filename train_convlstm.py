import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError

# Load preprocessed data
X_train = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/X_train.npy')
y_train = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/y_train.npy')
X_test = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/X_test.npy')
y_test = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/y_test.npy')

# Verify shapes
print("X_train shape:", X_train.shape)  # (6988, 24, 17, 25, 2)
print("y_train shape:", y_train.shape)  # (6988, 17, 25)

# Build ConvLSTM model
model = Sequential([
    ConvLSTM2D(32, (3, 3), activation='relu', input_shape=(24, 17, 25, 2),
               padding='same', return_sequences=False),
    Dense(1, activation='linear')  # Output: temperature grid (17×25)
])
model.compile(optimizer='adam', loss=MeanSquaredError())  # Explicit loss object
model.summary()

# Define EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',    # Monitor validation loss
    patience=10,           # Stop after 10 epochs with no improvement
    restore_best_weights=True,  # Use weights from the best epoch
    mode='min'             # Minimize loss
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on test set
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Save the model
model.save('C:/Users/91905/PycharmProjects/WeatherNWP/convlstm_weather_model.h5')