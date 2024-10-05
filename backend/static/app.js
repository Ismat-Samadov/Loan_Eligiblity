document.getElementById('submit').addEventListener('click', function () {
    // Prepare the input data
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

    // Make the API call to Flask
    fetch('https://loan-eligiblity.onrender.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(result => {
        // Display the prediction result
        const output = `
            <p>Loan Status: <strong>${result.loan_status}</strong></p>
            <p>Approval Probability: <strong>${(result.approval_probability * 100).toFixed(2)}%</strong></p>
        `;
        document.getElementById('result').innerHTML = output;
    })
    .catch(error => {
        // Handle errors in case of failed API request
        document.getElementById('result').innerHTML = `<p style="color:red">Error: ${error.message}</p>`;
    });
});
