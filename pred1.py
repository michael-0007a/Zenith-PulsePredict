import re
from flask import Flask, render_template, request, session, redirect, url_for
import os
import joblib
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
app.secret_key = 'zxgfchvjbklnfchvg'  

os.environ["GOOGLE_API_KEY"] = "AIzaSyAzM8A72SUVyqeGJ5o0rnV_Nt4dYX6ZUbM"  # Set this securely

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0, 
    max_tokens=None, 
    timeout=5, 
    max_retries=2
)


def load_model():
    clf = joblib.load('heart_disease_rf_model.joblib')
    return clf

clf = load_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None  
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        prediction = clf.predict(input_data)
        prediction_proba = clf.predict_proba(input_data)

        predicted_class = prediction[0]
        disease_probability = prediction_proba[0][predicted_class]

        if predicted_class == 1:
            result = f"Based on your input, you are predicted to have heart disease."
        else:
            result = f"Based on your input, you are predicted to not have heart disease."

        user_input_context = f"""
        Age: {age}
        Sex: {sex}
        Chest Pain Type: {cp}
        Resting Blood Pressure: {trestbps}
        Cholesterol: {chol}
        Fasting Blood Sugar: {fbs}
        Resting Electrocardiographic Results: {restecg}
        Maximum Heart Rate Achieved: {thalach}
        Exercise Induced Angina: {exang}
        Old Peak: {oldpeak}
        Slope of Peak Exercise ST Segment: {slope}
        Number of Major Vessels Colored by Fluoroscopy: {ca}
        Thalassemia: {thal}
        """

        session['user_input_context'] = user_input_context

    return render_template('predict.html', result=result)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    chatbot_response = None
    cleaned_response = ""  
    if 'chat_history' not in session:
        session['chat_history'] = []

    if 'user_input_context' in session:
        user_input_context = session['user_input_context']

        messages = [
            ("system", """You are a medical assistant who has 30 years of experience in Cardiology. You only answer queries related to cardiology and answer them in layman terms. If they are not medical queries say "I am a Cardiological Medical Assistant, ask me questions only related to cardiology." When the user gives their reports, if they have a risk of heart disease, you tell them what it can probably be, and tell them what can be done as immediate steps, and advise them to visit a cardiologist ASAP for further steps. Otherwise, give them precautions based on their reports."""),
            ("human", f"Here is some medical information:\n{user_input_context}\nCan you assist with further information?")
        ]

        ai_msg = llm.invoke(messages)
        chatbot_response = ai_msg.content  

        cleaned_response = re.sub(r'(\*\*)(.*?)\1', r'<strong>\2</strong>', chatbot_response)  # Bold
        cleaned_response = re.sub(r'(\*)(.*?)\1', r'<em>\2</em>', cleaned_response)  # Italic
        cleaned_response = re.sub(r'(`)(.*?)\1', r'<code>\2</code>', cleaned_response)  # Code
        cleaned_response = re.sub(r'([~])\s*', '', cleaned_response)  # Remove strikethrough if needed

        cleaned_response = re.sub(r'(\n)\s*', r'<br>', cleaned_response)  # Single line break to <br>
        cleaned_response = re.sub(r'(<br>)+', r'<br>', cleaned_response)  # Prevent multiple <br> from stacking
        cleaned_response = re.sub(r'(<br>){2,}', r'<br><br>', cleaned_response)  # Multiple breaks to a single double break

        session['chat_history'] = [{"role": "chatbot", "message": cleaned_response}]  # Clear old history and store the new response

    if request.method == 'POST':
        user_message = request.form['user_message']
        messages.append(("human", user_message))
        ai_msg = llm.invoke(messages)
        chatbot_response = ai_msg.content  # Get the response

        cleaned_response = re.sub(r'(\*\*)(.*?)\1', r'<strong></strong>', chatbot_response)  # Bold
        cleaned_response = re.sub(r'(\*)(.*?)\1', r'<em></em>', cleaned_response)  # Italic
        cleaned_response = re.sub(r'(`)(.*?)\1', r'<code></code>', cleaned_response)  # Code
        cleaned_response = re.sub(r'([~])\s*', '', cleaned_response)  # Remove strikethrough if needed

        session['chat_history'] = [{"role": "chatbot", "message": cleaned_response}]

    return render_template('chatbot.html', chatbot_response=cleaned_response, chat_history=session['chat_history'])


if __name__ == '__main__':
    app.run(debug=True)
