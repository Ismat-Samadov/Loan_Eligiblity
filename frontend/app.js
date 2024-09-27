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
        `;
        document.getElementById('result').innerHTML = output;
    })
    .catch(error => {
        document.getElementById('result').innerHTML = `<p style="color:red">Error: ${error.message}</p>`;
    });
});