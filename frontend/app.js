async function predictLoanStatus() {
    const data = {
        applicant_income: 5000,
        coapplicant_income: 2000,
        loan_amount: 150,
        loan_amount_term: 360,
        credit_history: 1,
        gender: "Male",
        married: "Yes",
        dependents: "0",
        education: "Graduate",
        self_employed: "No",
        property_area: "Urban"
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const result = await response.json();
        console.log(result);
    } catch (error) {
        console.error('Error:', error);
    }
}

predictLoanStatus();
