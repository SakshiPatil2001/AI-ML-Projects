import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the diabetes dataset
diabetes_data = pd.read_csv(r"C:\Users\saksh\OneDrive\Desktop\diabetes.csv")

# Display the first few rows of the dataset
print(diabetes_data.head())
# Assuming the target variable is 'Outcome' (as we are performing regression)
X = diabetes_data.drop('Outcome', axis=1)  # Features
y = diabetes_data['Outcome']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate XGBoost classifier
number_Boosts=5000
classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=42, n_estimators=number_Boosts)
#trying to get the number of ephochs right

# Train the model
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate the model (optional)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save predictions to a CSV file (optional)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('predictions.csv', index=False)
print(number_Boosts)

predicted_data = pd.read_csv('predictions.csv')
print(predicted_data)

print("Classification Report:")
print(classification_report(y_test, y_pred))