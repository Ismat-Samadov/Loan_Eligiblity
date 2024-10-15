# Loan Eligibility Prediction

This project aims to predict the eligibility of loan applicants based on various demographic and financial attributes. The dataset for this project is sourced from the [Analytics Vidhya Loan Prediction competition](https://www.kaggle.com/datasets/leonbora/analytics-vidhya-loan-prediction/data) on Kaggle.

## Project Overview

The goal is to build a predictive model that classifies whether an applicant will be eligible for a loan based on features such as gender, marital status, income, and credit history.

## Data Source

- The dataset used in this project can be found on Kaggle: [Loan Prediction Data](https://www.kaggle.com/datasets/leonbora/analytics-vidhya-loan-prediction/data).
  
## Repository Structure

- `data/`: Contains the dataset used for training the model.
- `notebooks/`: Jupyter notebooks with detailed exploratory data analysis (EDA), feature engineering, and model training.
- `src/`: Python scripts for data preprocessing, model training, and evaluation.
- `README.md`: Project overview and instructions.
  
## Data Preprocessing

- **Handling Missing Values**: Missing values were imputed using the mode for categorical variables and the mean for numerical variables.
- **Outlier Removal**: Outliers were detected and removed using the Interquartile Range (IQR) method.
- **Skewed Distribution**: Features like `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount` were transformed using the square root transformation to handle skewed distributions.
- **One-Hot Encoding**: Categorical variables were converted into numerical format using one-hot encoding, and unnecessary columns were dropped.

## Model

- The model used in this project is a **Random Forest Classifier** with hyperparameters tuned for optimal performance.
- The dataset had an imbalanced class distribution, and the **Synthetic Minority Over-sampling Technique (SMOTE)** was used to balance the classes.
- The model was evaluated using **accuracy**, **ROC-AUC score**, and a detailed **classification report**.

### Model Performance

- **Accuracy**: 83.72%
- **ROC-AUC**: 94.20%

## Feature Importance

The Random Forest model provided insights into the most important features influencing loan eligibility. A bar chart of feature importances was generated to visualize the contributions of different features.

## Installation and Usage

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ismat-Samadov/Loan_Eligiblity.git
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to train the model:
   ```bash
   jupyter notebook notebooks/Loan_Eligibility_Prediction.ipynb
   ```
   or
   ```bash
   python src/train_model.py
   ```

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib

## Contributing

Feel free to open an issue or submit a pull request if you would like to contribute to this project.
