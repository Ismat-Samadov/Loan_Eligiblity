from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model and scaler
def load_model():
    with open('model/best_rf_model.pkl', 'rb') as model_file, open('model/scaler.pkl', 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return model, scaler

# Predict loan status
def predict_loan_status(applicant_income, coapplicant_income, loan_amount, loan_amount_term, 
                        credit_history, gender, married, dependents, education, self_employed, property_area, 
                        model, scaler, threshold=0.5):

    # Adjust dependents encoding
    if dependents == '3':
        dependents = '3+'
    
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

    # Apply transformations
    input_data['ApplicantIncome'] = np.sqrt(input_data['ApplicantIncome'])
    input_data['CoapplicantIncome'] = np.sqrt(input_data['CoapplicantIncome'])
    input_data['LoanAmount'] = np.sqrt(input_data['LoanAmount'])
    
    # Apply one-hot encoding
    input_data = pd.get_dummies(input_data)

    # Expected columns
    expected_columns = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Gender', 'Married', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 
        'Education', 'Self_Employed', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
    ]

    # Add missing columns
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with 0

    # Reorder columns to match the model input
    input_data = input_data[expected_columns]
    
    # Normalize the input data using the saved MinMaxScaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]
    prediction = (prediction_prob >= threshold).astype(int)
    
    loan_status = "Approved" if prediction[0] == 1 else "Rejected"
    approval_probability = prediction_prob[0]
    
    return loan_status, approval_probability

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json
    applicant_income = data.get('applicant_income')
    coapplicant_income = data.get('coapplicant_income')
    loan_amount = data.get('loan_amount')
    loan_amount_term = data.get('loan_amount_term')
    credit_history = data.get('credit_history')
    gender = data.get('gender')
    married = data.get('married')
    dependents = data.get('dependents')
    education = data.get('education')
    self_employed = data.get('self_employed')
    property_area = data.get('property_area')

    # Load model and scaler
    model, scaler = load_model()

    # Call prediction function
    loan_status, approval_probability = predict_loan_status(
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_history, gender, married, dependents, education, self_employed, property_area,
        model, scaler
    )

    # Return the prediction, probability, and input data as the response
    return jsonify({
        "approval_probability": round(approval_probability, 3),
        "loan_status": loan_status,
        "input_data": {
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_amount_term": loan_amount_term,
            "credit_history": credit_history,
            "gender": gender,
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "property_area": property_area
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
