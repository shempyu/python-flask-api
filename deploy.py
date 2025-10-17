import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

def deploy_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        logger.info("✅ Model files already exist. Skipping training.")
        return

    logger.info("🚀 Setting up crop recommendation model...")

    df = pd.read_csv("Crop_recommendation.csv")
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    logger.info(f"✅ Model trained and saved!")
    logger.info(f"📊 Test accuracy: {model.score(scaler.transform(X_test), y_test):.4f}")

if __name__ == "__main__":
    deploy_model()
