import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/loan-train.csv')

# Step 1: Drop unnecessary variables (Loan_ID)
df = df.drop(['Loan_ID'], axis=1)

# Step 2: Data Imputation for Missing Values
# Imputation for Categorical Variables using mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

# Imputation for Numerical Variables using mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

# Step 3: One-hot Encoding for Categorical Variables
df = pd.get_dummies(df)

# Drop specific encoded columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis=1)

# Rename columns for clarity
df.rename(columns={'Gender_Male': 'Gender', 
                   'Married_Yes': 'Married', 
                   'Education_Graduate': 'Education', 
                   'Self_Employed_Yes': 'Self_Employed', 
                   'Loan_Status_Y': 'Loan_Status'}, inplace=True)

# Step 4: Remove Outliers Using IQR Method
numerical_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers based on IQR for numerical columns only
df = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 5: Skewed Distribution Treatment (Square Root Transformation)
df['ApplicantIncome'] = np.sqrt(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.sqrt(df['CoapplicantIncome'])
df['LoanAmount'] = np.sqrt(df['LoanAmount'])

# Step 6: Separate Features (X) and Target (y)
X = df.drop(['Loan_Status'], axis=1)
y = df['Loan_Status']

# Step 7: Apply SMOTE to handle class imbalance
X, y = SMOTE(random_state=42).fit_resample(X, y)

# Step 8: Data Normalization (MinMax Scaling)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train Random Forest Model with Best Hyperparameters
best_rf_model = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=45, 
    max_features='log2', 
    min_samples_leaf=1, 
    min_samples_split=2, 
    bootstrap=True,
    random_state=42,
    # max_leaf_nodes=25
)
best_rf_model.fit(X_train, y_train)

# Step 10: Model Evaluation
y_pred = best_rf_model.predict(X_test)
y_pred_prob = best_rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Print the best hyperparameters
print(f"Optimized Random Forest Accuracy: {accuracy:.4f}")
print(f"Optimized Random Forest ROC-AUC: {roc_auc:.4f}")
print(classification_report(y_test, y_pred))

# Step 11: Visualize Feature Importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = df.drop(columns=['Loan_Status']).columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title('Optimized Random Forest Feature Importance')
plt.tight_layout()
plt.show()
