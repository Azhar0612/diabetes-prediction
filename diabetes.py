# ==========================================
# Diabetes Prediction Using Logistic Regression (Final Fixed Version)
# ==========================================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ==========================================
# Step 2: Load Dataset
# ==========================================

data = pd.read_csv("diabetes.csv")

# ==========================================
# Step 3: Display Dataset
# ==========================================

print("========== FIRST 5 ROWS ==========")
print(data.head())

# ==========================================
# Step 4: Dataset Information
# ==========================================

print("\n========== DATASET INFORMATION ==========")
data.info()

print("\n========== DATASET DESCRIPTION ==========")
print(data.describe())

# ==========================================
# Step 5: Check Missing Values
# ==========================================

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# ==========================================
# Step 6: Data Cleaning (Fix applied)
# ==========================================

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    median_value = data[col].median()
    data[col] = data[col].replace(0, median_value)

print("\nData Cleaning Completed Successfully")

# ==========================================
# Step 7: Separate Features and Target
# ==========================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ==========================================
# Step 8: Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDataset Split Completed")
print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# ==========================================
# Step 9: Feature Scaling (Correct Placement)
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Step 10: Train Model
# ==========================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\nModel Training Completed Successfully")

# ==========================================
# Step 11: Test Model
# ==========================================

y_pred = model.predict(X_test_scaled)

# ==========================================
# Step 12: Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", round(accuracy, 3))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 13: Predict New Patient (ERROR FIXED)
# ==========================================

new_patient = pd.DataFrame(
    [[2, 180, 75, 30, 120, 28, 0.5, 45]],
    columns=X.columns
)

# Apply SAME scaling
new_patient_scaled = scaler.transform(new_patient)

prediction = model.predict(new_patient_scaled)

print("\n========== PREDICTION RESULT ==========")

if prediction[0] == 1:
    print("Diabetes: Yes")
else:
    print("Diabetes: No")

# ==========================================
# Step 14: Final Message
# ==========================================

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("Diabetes Prediction Model is Ready")
print("===================================")