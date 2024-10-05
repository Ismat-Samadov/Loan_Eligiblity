import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Set up Flask app with default 'static' and 'templates' folder
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the model and scaler
def load_model():
    with open('best_rf_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return model, scaler

# Convert and map the input data to the correct types and formats
def map_and_convert_input(data):
    mapped_data = {
        'ApplicantIncome': float(data['applicant_income']),
        'CoapplicantIncome': float(data['coapplicant_income']),
        'LoanAmount': float(data['loan_amount']),
        'Loan_Amount_Term': float(data['loan_amount_term']),
        'Credit_History': int(data['credit_history']),
        'Gender': data['gender'],
        'Married': data['married'],
        'Dependents': data['dependents'],
        'Education': data['education'],
        'Self_Employed': data['self_employed'],
        'Property_Area': data['property_area']
    }
    if mapped_data['Dependents'] == '3':
        mapped_data['Dependents'] = '3+'
    
    return mapped_data

# Predict loan status using the saved model
def predict_loan_status(data, model, scaler, threshold=0.5):
    input_data = pd.DataFrame([data])
    input_data['ApplicantIncome'] = np.sqrt(input_data['ApplicantIncome'])
    input_data['CoapplicantIncome'] = np.sqrt(input_data['CoapplicantIncome'])
    input_data['LoanAmount'] = np.sqrt(input_data['LoanAmount'])

    input_data = pd.get_dummies(input_data)

    expected_columns = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Gender', 'Married', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
        'Education', 'Self_Employed', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
    ]
    
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[expected_columns]
    input_data_scaled = scaler.transform(input_data)

    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]
    prediction = (prediction_prob >= threshold).astype(int)
    
    loan_status = "Approved" if prediction[0] == 1 else "Rejected"
    approval_probability = prediction_prob[0]
    
    return loan_status, approval_probability

# API Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    mapped_data = map_and_convert_input(data)
    model, scaler = load_model()
    loan_status, approval_probability = predict_loan_status(mapped_data, model, scaler)

    return jsonify({
        "approval_probability": round(approval_probability, 3),
        "loan_status": loan_status,
        "input_data": mapped_data
    })

# Serve index.html at root URL
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
