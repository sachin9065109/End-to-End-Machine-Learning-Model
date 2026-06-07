import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Print columns for debugging
print("Available Columns:")
print(df.columns.tolist())

# Check required columns
required_columns = ["Age", "Income", "Loan"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in CSV file.\n"
            f"Available columns are: {df.columns.tolist()}"
        )

# Features and target
X = df[["Age", "Income"]]
y = df["Loan"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
