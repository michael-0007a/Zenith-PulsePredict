# Zenith - PulsePredict


## Overview
This project consists of three components:

1. Heart Disease Prediction: A machine learning model built using Random Forest to predict the likelihood of heart disease based on patient data.
2. Medical Chatbot: A chatbot powered by Google's Gemini API, acting as a virtual medical assistant specializing in cardiology.
3. Web Interface: A Flask-based web application providing an interface for users to interact with the prediction model and chatbot.

## Features
### 1. Heart Disease Prediction
- Algorithm: Random Forest Classifier.
- Input: Patient data including age, sex, cholesterol levels, resting blood pressure, and more.
- Output: Predicts whether the user has a risk of heart disease, along with the probability score.
- Model Training: Hyperparameter tuning performed to achieve optimal accuracy.

### 2. Medical Chatbot
- Model: Google Gemini-1.5 API integration via LangChain.
- Functionality: 
  - Responds to cardiology-related queries in layman terms.
  - Provides immediate steps and precautions based on prediction results.
  - Rejects non-medical queries.

### 3. Web Interface
- Home Page: Introduction to the application.
- Prediction Page: Input form for patient data and displays prediction results.
- Chatbot Page: Interactive chatbot to guide users based on their reports and general cardiology-related questions.


## Installation and Usage

### 1. Prerequisites
- Python 3.7+
- Google Cloud account with API key for Gemini API.
- Required Python libraries:
  - Flask
  - Pandas
  - Scikit-learn
  - Joblib
  - LangChain Google Generative AI
  - Matplotlib, Seaborn (optional for testing)
- Web browser for interacting with the app.

### 2. Installation
1. Clone the repository:
   
2. Install dependencies:
  - Flask
  - Pandas
  - Scikit-learn
  - Joblib
  - LangChain Google Generative AI
  - Matplotlib, Seaborn (optional for testing)
   
3. Set up the Google API Key:

   Replace your_google_api_key with your Google Cloud API key.

4. Place the heart_disease_rf_model.joblib file in the root directory.


## Running the Application
1. Start the Flask server:
   python pred1.py
   
2. Open a web browser and navigate to http://127.0.0.1:5000/.


## File Structure

|-- app.py               # Script for training and saving the Random Forest model
|-- pred1.py             # Flask application for web interface and chatbot
|-- test.py              # Script for testing and hyperparameter tuning
|-- templates/           # HTML files for the web app
    |-- home.html
    |-- predict.html
    |-- chatbot.html
|-- static/              # CSS and JavaScript files
|-- heart.csv            # Dataset used for training
|-- heart_disease_rf_model.joblib  # Pre-trained Random Forest model


## Usage
1. Navigate to the Prediction Page to input patient data and get a prediction.
2. Use the Chatbot Page to ask cardiology-related queries based on the prediction or general health concerns.

## Results and Accuracy
- Achieved 95% accuracy on the test dataset using the Random Forest model.
- Integrated Google Gemini API for enhanced chatbot functionality.


## Future Enhancements
- Add more advanced preprocessing pipelines.
- Expand chatbot functionality to cover other medical domains.
- Incorporate user authentication for saving reports.

## Author
Developed by Michael Benedict (https://github.com/michael-0007a).  
