"""
train_model.py

This just trains the churn model and saves it so the streamlit app can load it later.
Run this once before running the app (or before deploying).

Dataset: Customer Churn Dataset
https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
(same dataset used in Assignment 8, model_optimization.ipynb)

Put the csv (customer_churn_dataset-training-master.csv, or renamed to churn_train.csv)
in the same folder as this script before running.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
SAMPLE_SIZE = 20000  # same sample size used in assignment 8, keeps training fast

print("loading data...")
df = pd.read_csv("churn_train.csv")
df = df.dropna()

# same sampling as assignment 8, so the results/insights stay consistent
df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
df_model = df_sample.drop(columns=["CustomerID"])

cat_cols = ["Gender", "Subscription Type", "Contract Length"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("training random forest (using the best params found in assignment 8's GridSearchCV)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_STATE,
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Test F1 Score:", round(f1_score(y_test, y_pred), 4))

# saving everything the app needs to make predictions later
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(feature_columns, "model/feature_columns.pkl")

print("done! saved model + scaler + encoders + feature_columns to the model/ folder")
