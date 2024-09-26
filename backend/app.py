from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
model_path = 'model/best_rf_model.pkl'
scaler_path = 'model/scaler.pkl'

def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load model and scaler
model = load_model(model_path)
scaler = load_model(scaler_path)

# Prediction function
def predict_loan_status(applicant_income, coapplicant_income, loan_amount, loan_amount_term, 
                        credit_history, gender, married, dependents, education, self_employed, property_area, 
                        threshold=0.5):
    # Map the dependents value to be consistent with training data
    if dependents == '3':
        dependents = '3+'  # Match the training data encoding
    
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'Property_Area': [property_area]
    })

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
    
    # Normalize the input data using MinMaxScaler
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction and check probabilities
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]  # Get probability of being "Approved"
    
    # Apply a custom threshold to the probability
    prediction = (prediction_prob >= threshold).astype(int)
    
    # Return both the loan status and probability
    loan_status = "Approved" if prediction[0] == 1 else "Rejected"
    return loan_status, prediction_prob[0]

# API endpoint to predict loan status
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract input values from JSON request
    applicant_income = data['ApplicantIncome']
    coapplicant_income = data['CoapplicantIncome']
    loan_amount = data['LoanAmount']
    loan_amount_term = data['Loan_Amount_Term']
    credit_history = data['Credit_History']
    gender = data['Gender']
    married = data['Married']
    dependents = data['Dependents']
    education = data['Education']
    self_employed = data['Self_Employed']
    property_area = data['Property_Area']
    
    # Predict loan status and probability
    loan_status, prob = predict_loan_status(applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                                            credit_history, gender, married, dependents, education, 
                                            self_employed, property_area)
    
    # Return the prediction and probability as a JSON response
    return jsonify({
        'loan_status': loan_status,
        'approval_probability': round(prob, 4)  # Rounded to 4 decimal places
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
