import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Ld dataset
df = pd.read_csv("data.csv")

df.columns = df.columns.str.strip()

# Print clm for dbg
print("Available Columns:")
print(df.columns.tolist())

required_columns = ["Age", "Income", "Loan"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in CSV file.\n"
            f"Available columns are: {df.columns.tolist()}"
        )

X = df[["Age", "Income"]]
y = df["Loan"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")
