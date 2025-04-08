from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.losses import MeanSquaredError
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model and scaler
try:
    model = load_model('C:/Users/91905/PycharmProjects/WeatherNWP/convlstm_weather_model.h5',
                       custom_objects={'MeanSquaredError': MeanSquaredError})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open('C:/Users/91905/PycharmProjects/WeatherNWP/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

# Load full dataset for context (using X_train as a proxy)
try:
    X_full = np.load('C:/Users/91905/PycharmProjects/WeatherNWP/X_train.npy')
    print("X_full loaded successfully. Shape:", X_full.shape)
except Exception as e:
    print(f"Error loading X_full: {e}")

@app.route('/', methods=['GET', 'POST'])
def predict():
    image_path = None
    if request.method == 'POST':
        # Get date and time from form
        date_str = request.form['date']
        time_str = request.form['time']
        try:
            input_date = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
        except ValueError:
            return "Invalid date/time format. Use YYYY-MM-DD HH:MM."

        # Define the range of trained data (2023) and allow future predictions
        start_date = datetime(2023, 1, 1, 0, 0)
        end_date = datetime(2023, 12, 31, 23, 0)
        last_sequence_date = end_date  # Last trained date
        if input_date < start_date:
            return "Date must be on or after 2023-01-01 00:00."

        # Calculate index for trained data, use last sequence for future predictions
        idx = int((input_date - start_date).total_seconds() / 3600) - 24
        if 0 <= idx < len(X_full):
            X_input = X_full[idx:idx+1]  # Use actual sequence for 2023
        else:
            # For future dates, reuse the last sequence (index 6987)
            idx = len(X_full) - 1
            X_input = X_full[idx:idx+1]
            print(f"Using last trained sequence for future prediction at {input_date}")

        # Predict
        y_pred = model.predict(X_input)

        # Inverse scale
        y_pred_reshaped = y_pred.squeeze(-1).reshape(-1, 1)
        dummy = np.zeros((y_pred_reshaped.shape[0], 1))
        y_pred_unscaled = scaler.inverse_transform(np.hstack([y_pred_reshaped, dummy]))[:, 0]
        prediction = y_pred_unscaled.reshape(17, 25)

        # Generate heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(prediction, cmap='coolwarm')
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Predicted Temperature for {date_str} {time_str}')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')
        image_path = os.path.join(app.root_path, 'static', 'prediction_heatmap.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()

    return render_template('index.html', image_path=image_path)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Use waitress for deployment
    from waitress import serve
    print("Starting waitress server on 0.0.0.0:5000...")
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Server IP: {local_ip}. Access via http://{local_ip}:5000/ or http://127.0.0.1:5000/")
    try:
        serve(app, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")