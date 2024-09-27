document.getElementById('submit').addEventListener('click', function () {
    const data = {
        applicant_income: parseFloat(document.getElementById('applicant_income').value),
        coapplicant_income: parseFloat(document.getElementById('coapplicant_income').value),
        loan_amount: parseFloat(document.getElementById('loan_amount').value),
        loan_amount_term: parseFloat(document.getElementById('loan_amount_term').value),
        credit_history: parseInt(document.getElementById('credit_history').value),
        gender: document.getElementById('gender').value,
        married: document.getElementById('married').value,
        dependents: document.getElementById('dependents').value,
        education: document.getElementById('education').value,
        self_employed: document.getElementById('self_employed').value,
        property_area: document.getElementById('property_area').value
    };

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        const output = `
            <p>Loan Status: <strong>${result.loan_status}</strong></p>
            <p>Approval Probability: <strong>${(result.approval_probability * 100).toFixed(2)}%</strong></p>
            <h3>Input Data</h3>
            <ul>
                <li>Applicant Income: ${result.input_data.ApplicantIncome}</li>
                <li>Coapplicant Income: ${result.input_data.CoapplicantIncome}</li>
                <li>Loan Amount: ${result.input_data.LoanAmount}</li>
                <li>Loan Amount Term: ${result.input_data.Loan_Amount_Term}</li>
                <li>Credit History: ${result.input_data.Credit_History}</li>
                <li>Gender: ${result.input_data.Gender}</li>
                <li>Married: ${result.input_data.Married}</li>
                <li>Dependents: ${result.input_data.Dependents}</li>
                <li>Education: ${result.input_data.Education}</li>
                <li>Self Employed: ${result.input_data.Self_Employed}</li>
                <li>Property Area: ${result.input_data.Property_Area}</li>
            </ul>
        `;
        document.getElementById('result').innerHTML = output;
    })
    .catch(error => {
        document.getElementById('result').innerHTML = `<p style="color:red">Error: ${error.message}</p>`;
    });
});