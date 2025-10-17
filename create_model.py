#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

def create_crop_model():
    """Create and save the crop recommendation model"""
    
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        logger.info("✅ Model files already exist. Skipping training.")
        return True

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    crop_dict_reverse = {v.lower(): k for k, v in crop_dict.items()}

    try:
        logger.info("📁 Loading dataset...")
        df = pd.read_csv("Crop_recommendation.csv")

        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label'].map(crop_dict_reverse)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)

        logger.info(f"✅ Model trained and saved: {MODEL_FILE}, {SCALER_FILE}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        return False

if __name__ == "__main__":
    success = create_crop_model()
    if success:
        logger.info("🎉 Model creation completed successfully!")
    else:
        logger.error("💥 Model creation failed!")
