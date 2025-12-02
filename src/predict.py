import pickle
import numpy as np

model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_transaction(features):
    features = np.array(features).reshape(1, -1)
    features[:, :2] = scaler.transform(features[:, :2])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return prediction, probability
