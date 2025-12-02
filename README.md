# Credit Card Fraud Detection

A machine learning model that detects fraudulent credit card transactions using PCA-transformed features.
The project includes a Streamlit interface where users can input feature values and receive instant predictions.

---

## Overview

Credit card fraud is a significant challenge in the digital payment ecosystem.
This project builds a binary classification model to identify fraudulent transactions using the Kaggle Credit Card Fraud Dataset (2013).

Because the dataset is highly imbalanced, special handling techniques were applied to ensure reliable fraud detection.

The final workflow includes:

* Data preprocessing
* Class imbalance handling
* Model training and evaluation
* Model serialization
* Streamlit web application

---

## Project Structure

```
Credit-Card-Fraud-Detection/
│── data/
│   └── creditcard.csv
│
│── model/
│   └── fraud_model.pkl
│
│── src/
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
│── requirements.txt
│── README.md
│── .gitignore
```

---

## Dataset

The dataset contains credit card transactions made by European cardholders in 2013.

Key points:

* 284,807 total transactions
* 492 fraud cases (0.172 percent)
* Strongly imbalanced
* Features V1–V28 are anonymized PCA components
* Only `Time` and `Amount` are original features
* Target variable: `Class` (0 = legitimate, 1 = fraud)

Dataset source:
[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## Model Development

### 1. Preprocessing

* Scaled the `Amount` feature with StandardScaler
* Dropped the `Time` feature
* PCA-transformed features (V1–V28) were kept as provided

### 2. Handling Class Imbalance

Given the extreme imbalance, several approaches were tested:

* Random undersampling
* Random oversampling
* SMOTE (Synthetic Minority Oversampling Technique)

SMOTE produced the most balanced recall and precision values and was selected.

### 3. Model Training

Multiple algorithms were evaluated:

| Model               | Accuracy  | Precision | Recall   | ROC-AUC |
| ------------------- | --------- | --------- | -------- | ------- |
| Logistic Regression | Good      | Good      | Moderate | Good    |
| Random Forest       | Very High | Very High | Moderate | High    |
| XGBoost             | High      | High      | High     | Highest |

XGBoost delivered the best tradeoff between false positives and false negatives.

### 4. Model Saving

The trained model is saved as:

```
fraud_model.pkl
```

This file is excluded from GitHub via `.gitignore`.

---

## Streamlit Application

The user interface allows the following:

* Input of PCA-based features (V1–V28)
* Model prediction on button click
* Display of prediction with the associated probability
* Clear distinction between legitimate and fraudulent transactions

---

## Running the Project Locally

### 1. Clone the repository

```
git clone https://github.com/rizwanalimondal/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install the required packages

```
pip install -r requirements.txt
```

### 3. Start the application

```
streamlit run src/app.py
```

---

## Deployment

This project can be deployed on:

* Streamlit Cloud
* Render
* HuggingFace Spaces

Only the app script and the serialized model are necessary for deployment.

The project has been deployed on Streamlit Cloud and can be accessed from below:

https://credit-card-fraud-detection-rqzyjzkqiq9fmumqzreql7.streamlit.app/

---

## Future Improvements

* Introduce SHAP explainability to understand model behavior
* Explore Autoencoders or Isolation Forest for anomaly detection
* Add batch prediction endpoints
* Add monitoring for drift or model decay

---

## License

MIT License.

---

## Acknowledgements

Dataset: Kaggle Credit Card Fraud Detection Dataset (2013)
Modeling and implementation: Rizwan Ali Mondal

---
