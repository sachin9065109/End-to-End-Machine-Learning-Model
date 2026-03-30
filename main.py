import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data.csv")

# Features & target
X = df[["Age", "Income"]]
y = df["Loan"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")