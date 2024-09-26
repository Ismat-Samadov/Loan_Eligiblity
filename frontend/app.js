document.getElementById('loanForm').addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent the form from submitting the default way

    // Gather form data
    const formData = {
        ApplicantIncome: document.getElementById('applicantIncome').value,
        CoapplicantIncome: document.getElementById('coapplicantIncome').value,
        LoanAmount: document.getElementById('loanAmount').value,
        Loan_Amount_Term: document.getElementById('loanAmountTerm').value,
        Credit_History: document.getElementById('creditHistory').value,
        Gender: document.getElementById('gender').value,
        Married: document.getElementById('married').value,
        Dependents: document.getElementById('dependents').value,
        Education: document.getElementById('education').value,
        Self_Employed: document.getElementById('selfEmployed').value,
        Property_Area: document.getElementById('propertyArea').value
    };

    // Send the data to the backend via POST request
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
        .then(response => response.json())
        .then(data => {
            // Display the result
            document.getElementById('result').innerHTML = `
                <p>Loan Status: <strong>${data.loan_status}</strong></p>
                <p>Approval Probability: <strong>${(data.approval_probability * 100).toFixed(2)}%</strong></p>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
        });
});
