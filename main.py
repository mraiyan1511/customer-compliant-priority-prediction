import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# CUSTOMER COMPLAINT PRIORITY PREDICTION
# USING RANDOM FOREST
# ==========================================

# Load Dataset
df = pd.read_csv("complaints.csv")

print("Original Data Loaded:", df.shape)

# ==========================================
# DATA CLEANING
# ==========================================

df = df.drop_duplicates()

# Create Priority
# High priority:
# - Consumer disputed = Yes
# - Timely response = No
#
# Medium priority:
# - Other complaints

df["Priority"] = "Medium"

df.loc[
    (df["Consumer disputed?"] == "Yes") |
    (df["Timely response?"] == "No"),
    "Priority"
] = "High"

# Fill missing values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].fillna("Unknown")

print("Priority Created:")
print(df["Priority"].value_counts())

# ==========================================
# SELECT USEFUL FEATURES
# ==========================================

features = [
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?"
]

X = df[features].copy()
y = df["Priority"]

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

label_encoders = {}

for column in X.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column].astype(str)
    )

    label_encoders[column] = encoder

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

print("\nFeatures:")
print(X.head())

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("CUSTOMER COMPLAINT PRIORITY PREDICTION")
print("===================================")

print(
    "Random Forest Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# ==========================================
# SAVE EVERYTHING
# ==========================================

model_data = {
    "model": model,
    "encoders": label_encoders,
    "target_encoder": target_encoder,
    "features": features
}

joblib.dump(
    model_data,
    "customer_complaint_priority_model.pkl"
)

print("\nModel saved successfully!")
print("File: customer_complaint_priority_model.pkl")