import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the model and scaler
def load_model():
    with open('model/best_rf_model.pkl', 'rb') as model_file, open('model/scaler.pkl', 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return model, scaler

# Convert and map the input data to the correct types and formats
def map_and_convert_input(data):
    # Convert numeric fields to float/int
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
    
    # If Dependents is '3', we convert it to '3+' as in the training data
    if mapped_data['Dependents'] == '3':
        mapped_data['Dependents'] = '3+'
    
    return mapped_data

# Predict loan status using the saved model
def predict_loan_status(data, model, scaler, threshold=0.7):
    # Convert input data
    input_data = pd.DataFrame([data])

    # Apply transformations used during training
    input_data['ApplicantIncome'] = np.sqrt(input_data['ApplicantIncome'])
    input_data['CoapplicantIncome'] = np.sqrt(input_data['CoapplicantIncome'])
    input_data['LoanAmount'] = np.sqrt(input_data['LoanAmount'])

    # Apply one-hot encoding (must be the same as during training)
    input_data = pd.get_dummies(input_data)

    # Add missing columns that were present during training
    expected_columns = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Gender', 'Married', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
        'Education', 'Self_Employed', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
    ]
    
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with 0
    
    # Reorder columns to match the model input
    input_data = input_data[expected_columns]

    # Normalize the input data using the saved MinMaxScaler
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction and check probabilities
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]  # Get probability of being "Approved"
    prediction = (prediction_prob >= threshold).astype(int)
    
    loan_status = "Approved" if prediction[0] == 1 else "Rejected"
    approval_probability = prediction_prob[0]
    
    return loan_status, approval_probability

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json

    # Map and convert input data
    mapped_data = map_and_convert_input(data)

    # Load model and scaler
    model, scaler = load_model()

    # Call prediction function
    loan_status, approval_probability = predict_loan_status(mapped_data, model, scaler)

    # Return the prediction, probability, and input data as the response
    return jsonify({
        "approval_probability": round(approval_probability, 3),  # Round to 3 decimal places
        "loan_status": loan_status,
        "input_data": mapped_data
    })

if __name__ == '__main__':
    app.run(debug=True)
