import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import xgboost as xgb
import time 



# Load the lead conversion dataset
data = r"c:\Users\saksh\Downloads\CRM_LeadsData .csv"
augmented_data= r"C:\Users\saksh\Downloads\augmented_data.csv"
lead_data = pd.read_csv(augmented_data)

#Data preprocess
# Drop rows with any null values
lead_data.dropna(axis=1, how='all', inplace=True)

# filling null values in Target variable with 0
lead_data['cRMLeadStatusMasterName'].fillna('0', inplace=True)

# Separate features (X) and target variable (y)
X = lead_data.drop("cRMLeadStatusMasterName", axis=1)  # Features
y = lead_data["cRMLeadStatusMasterName"]               # Target variable

print(y)
 
 
# Define the mapping dictionary for target variable
target_mapping = {"Did not connect": 0,
                  'Business profile sent': 1,
                  'Follow up to share business profile': 1,
                  'Follow up': 1,
                  'Warm lead': 1,
                  'Not Qualified': 0,
                  '0': 0}


y = y.map(target_mapping, na_action='ignore')

print(y)
print(target_mapping)
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
classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(pipeline, "lead_status_prediction_model.pkl")

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)                   



#Test Set-

# Create a DataFrame for results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Save results to CSV
results_df.to_csv("test_results.csv", index=False)