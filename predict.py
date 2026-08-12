import pandas as pd
import joblib

# Load trained model
model_data = joblib.load(
    "customer_complaint_priority_model.pkl"
)

model = model_data["model"]
encoders = model_data["encoders"]
target_encoder = model_data["target_encoder"]
features = model_data["features"]

# Load test data
data_df = pd.read_csv("complaint_test_data.csv")
print("Test Data Loaded:", data_df.shape)
print("Test Data Columns:")
print(data_df.columns.tolist())

# Check missing columns
missing_columns = [
    col for col in features
    if col not in data_df.columns
]

if missing_columns:
    print("\nERROR: Missing columns in test_data.csv:")
    print(missing_columns)
    print("\nYour test_data.csv must contain these columns:")
    print(features)
    exit()

# Select features
X = data_df[features].copy()

# Fill missing values
for column in X.select_dtypes(include="object").columns:
    X[column] = X[column].fillna("Unknown")
    X[column] = X[column].astype(str)

# Encode categorical columns
for column in X.select_dtypes(include="object").columns:

    encoder = encoders[column]

    known_values = set(encoder.classes_)

    X[column] = X[column].apply(
        lambda value: value
        if value in known_values
        else encoder.classes_[0]
    )

    X[column] = encoder.transform(X[column])

# Prediction
y_pred = model.predict(X)

# Convert prediction to High / Medium
predicted_priority = target_encoder.inverse_transform(y_pred)

# Add prediction
data_df["Predicted Priority"] = predicted_priority

print("\n===================================")
print("CUSTOMER COMPLAINT PRIORITY")
print("===================================")

print(data_df["Predicted Priority"])

# Save result
data_df.to_csv(
    "prediction_results.csv",
    index=False
)

print("\nPrediction completed successfully!")
print("Result saved as: prediction_results.csv")