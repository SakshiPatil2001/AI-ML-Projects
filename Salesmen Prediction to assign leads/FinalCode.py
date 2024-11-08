import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import joblib

# Load leads data
leads_data = pd.read_csv(r'C:\Users\saksh\OneDrive\Desktop\Salesman prediction\leads_data.csv')

# Load salesman data
salesman_data = pd.read_csv(r'C:\Users\saksh\OneDrive\Desktop\Salesman prediction\salesmen_data.csv')

# Merge data based on Assigned_Salesman_ID
merged_data = pd.merge(leads_data, salesman_data, left_on='Assigned_Salesman_ID', right_on='Salesman_ID', how='left')

# Adjust Assigned_Salesman_ID to start from 0
merged_data['Assigned_Salesman_ID'] -= 1

# Separate features (X) and target variable (y)
X = merged_data.drop(['Lead_ID', 'Assigned_Salesman_ID', 'Salesman_ID'], axis=1)
y = merged_data['Assigned_Salesman_ID']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the XGBoost classifier
classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=10, random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate accuracy and F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Output predictions for each lead
lead_salesman_assignments = pd.DataFrame({'Lead_ID': X_test.index, 'Assigned_Salesman_ID': y_pred + 1})

# Merge predictions with salesman data to get detailed assignments
detailed_assignments = pd.merge(lead_salesman_assignments, salesman_data, left_on='Assigned_Salesman_ID', right_on='Salesman_ID')

# Save detailed assignments to CSV
detailed_assignments.to_csv(r'C:\Users\saksh\OneDrive\Desktop\Salesman prediction\detailed_lead_salesman_assignments.csv', index=False)


# Load the saved predictions
predictions = pd.read_csv(r'C:\Users\saksh\OneDrive\Desktop\Salesman prediction\lead_salesman_assignments.csv')

# Display the predictions
print(predictions)

# Load salesman data
salesman_data = pd.read_csv(r'C:\Users\saksh\OneDrive\Desktop\Salesman prediction\salesmen_data.csv')

# Merge predictions with salesman data to get the details of assigned salesmen
detailed_assignments = pd.merge(predictions, salesman_data, left_on='Assigned_Salesman_ID', right_on='Salesman_ID')

# Display the detailed assignments
print(detailed_assignments)


print("Detailed lead-salesman assignments predicted and saved.")
