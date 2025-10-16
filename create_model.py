#!/usr/bin/env python3
"""
Model Creation Script for Crop Recommendation API
Run this script first to generate model.pkl and scaler.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def create_crop_model():
    """Create and save the crop recommendation model"""
    
    print("🚀 Starting Crop Recommendation Model Creation...")
    
    # Crop dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    
    # Reverse dictionary for encoding
    crop_dict_reverse = {v.lower(): k for k, v in crop_dict.items()}
    
    try:
        # Load dataset
        print("📁 Loading dataset...")
        df = pd.read_csv("Crop_recommendation.csv")
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Display dataset info
        print(f"📊 Dataset columns: {list(df.columns)}")
        print(f"🌾 Crop types: {df['label'].nunique()} unique crops")
        print("📋 Crop distribution:")
        print(df['label'].value_counts())
        
        # Prepare features and target
        print("🔧 Preparing features and target...")
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label'].map(crop_dict_reverse)
        
        # Display feature statistics
        print("\n📈 Feature Statistics:")
        print(X.describe())
        
        # Split data
        print("\n🎯 Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📚 Training set: {X_train.shape[0]} samples")
        print(f"🧪 Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("⚖️ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        print("🤖 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        print(f"📊 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Feature Importance:")
        print(feature_importance)
        
        # Save model and scaler
        print("\n💾 Saving model and scaler...")
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        # Verify files are created
        if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
            print("✅ Model files created successfully!")
            print("   - model.pkl (Trained Random Forest model)")
            print("   - scaler.pkl (Feature scaler)")
        else:
            print("❌ Error: Model files were not created!")
            return False
        
        # Test prediction with sample data
        print("\n🧪 Testing model with sample data...")
        sample_features = np.array([[65, 37, 40, 23.36, 83.60, 5.33, 188.41]])
        sample_scaled = scaler.transform(sample_features)
        prediction = model.predict(sample_scaled)[0]
        probability = np.max(model.predict_proba(sample_scaled))
        
        print(f"🌱 Sample Prediction: {crop_dict[prediction]} (Confidence: {probability*100:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return False

if __name__ == '__main__':
    success = create_crop_model()
    
    if success:
        print("\n🎉 Model creation completed successfully!")
        print("🚀 You can now run: python app.py")
    else:
        print("\n💥 Model creation failed!")
        print("🔧 Please check your dataset and try again.")