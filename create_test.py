import pandas as pd

# Load customer complaint dataset
df = pd.read_csv("complaints.csv")

# Same features used in main.py
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

# Take 2 customer complaints
test_data = df[features].head(2)

# Save test data
test_data.to_csv("complaint_test_data.csv", index=False)

print("Test data created successfully!")
print(test_data)