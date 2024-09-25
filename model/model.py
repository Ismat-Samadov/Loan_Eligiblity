import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
loan_train = pd.read_csv('data/loan-train.csv')

# Step 1: Clean the data
# Handling missing values: Fill missing LoanAmount with median, Loan_Amount_Term with mode, and Credit_History with mode
loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].median(), inplace=True)
loan_train['Loan_Amount_Term'].fillna(loan_train['Loan_Amount_Term'].mode()[0], inplace=True)
loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode()[0], inplace=True)

# Encoding categorical variables manually using OneHotEncoder
loan_train['Gender'] = loan_train['Gender'].fillna('Unknown')
loan_train['Married'] = loan_train['Married'].fillna('Unknown')
loan_train['Dependents'] = loan_train['Dependents'].replace('3+', '3')
loan_train['Dependents'] = loan_train['Dependents'].fillna('0')
loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna('No')

# Use OneHotEncoder to handle categorical columns like Gender, Married, Dependents, Education, Self_Employed, Property_Area
onehot_encoder = OneHotEncoder(sparse_output=False)  # Fix: use sparse_output

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
encoded_cols = onehot_encoder.fit_transform(loan_train[categorical_columns])

# Convert encoded columns to a DataFrame and concatenate to the original dataframe
encoded_df = pd.DataFrame(encoded_cols, columns=onehot_encoder.get_feature_names_out(categorical_columns))
loan_train = pd.concat([loan_train, encoded_df], axis=1)

# Drop the original categorical columns
loan_train.drop(columns=categorical_columns, inplace=True)

# Encoding Loan_Status as it is the target variable
label_encoder = LabelEncoder()
loan_train['Loan_Status'] = label_encoder.fit_transform(loan_train['Loan_Status'])

# Step 2: Prepare the data
X = loan_train.drop(columns=['Loan_ID', 'Loan_Status'])
y = loan_train['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply LazyPredict
clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display models sorted by accuracy
models_sorted = models.sort_values(by="Accuracy", ascending=False)

# Step 4: Visualize the results
plt.figure(figsize=(10, 6))
sns.barplot(x=models_sorted.index, y=models_sorted['Accuracy'])
plt.xticks(rotation=90)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.tight_layout()
plt.show()

# Print out the sorted model results
print(models_sorted)
