import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# Load Data
df = pd.read_csv("../data/creditcard.csv")

# Features & Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale "Amount" and "Time"
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, y_pred))

# Save Model + Scaler
pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
