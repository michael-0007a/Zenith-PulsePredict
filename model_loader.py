import os
import joblib
import requests
from io import BytesIO

def load_model():
    # Check if model exists locally
    if not os.path.exists('heart_disease_rf_model.joblib'):
        print("Downloading model from GitHub...")
        # Replace with your GitHub raw file URL after uploading
        model_url = "https://raw.githubusercontent.com/michael-0007a/Zenith-PulsePredict/main/heart_disease_rf_model.joblib"
        
        try:
            # Download the model
            response = requests.get(model_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Save the model locally
            with open('heart_disease_rf_model.joblib', 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    # Load and return the model
    return joblib.load('heart_disease_rf_model.joblib')