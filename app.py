from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Function to train and save model
def train_and_save_model():
    from deploy import create_crop_model  # or copy the function code here
    success = create_crop_model()
    if success:
        logger.info("✅ Model trained and saved successfully.")
    else:
        logger.error("❌ Failed to create model.")
        raise RuntimeError("Model training failed.")

    global model, scaler
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

# Try loading model and scaler, train if missing
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    logger.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ Model files not found or failed to load: {e}")
    train_and_save_model()

@app.route('/')
def home():
    return jsonify({
        "message": "🌱 Crop Recommendation API is running!",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        features = [data[field] for field in required_fields]
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        crop_code = int(prediction[0])
        crop_name = crop_dict.get(crop_code, "Unknown Crop")

        return jsonify({
            "success": True,
            "recommended_crop": crop_name,
            "crop_code": crop_code,
            "input_parameters": data
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
