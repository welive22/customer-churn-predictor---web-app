"""
app.py - Streamlit app for Assignment 9

Loads the Random Forest churn model I trained in Assignment 8 (train_model.py just
retrains + saves it so this app can load it) and lets a user punch in some customer
details to see if the model thinks they're gonna churn or not.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---- loading the saved model stuff ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return model, scaler, encoders, feature_columns

model, scaler, encoders, feature_columns = load_artifacts()

st.title("📉 Customer Churn Predictor")
st.write(
    "This app uses the Random Forest model I trained in Assignment 8 to predict "
    "whether a customer is likely to churn (leave) or not, based on their info. "
    "Just fill in the details on the left and hit predict!"
)

# ---- sidebar inputs ----
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 65, 35)
gender = st.sidebar.selectbox("Gender", options=list(encoders["Gender"].classes_))
tenure = st.sidebar.slider("Tenure (months with the company)", 1, 60, 24)
usage_freq = st.sidebar.slider("Usage Frequency (times used per month)", 1, 30, 15)
support_calls = st.sidebar.slider("Support Calls (last period)", 0, 10, 2)
payment_delay = st.sidebar.slider("Payment Delay (days)", 0, 30, 5)
subscription_type = st.sidebar.selectbox("Subscription Type", options=list(encoders["Subscription Type"].classes_))
contract_length = st.sidebar.selectbox("Contract Length", options=list(encoders["Contract Length"].classes_))
total_spend = st.sidebar.slider("Total Spend ($)", 100, 1000, 500)
last_interaction = st.sidebar.slider("Last Interaction (days ago)", 1, 30, 10)

predict_btn = st.sidebar.button("Predict Churn", type="primary")

# ---- building the input row for the model ----
def build_input_df():
    row = {
        "Age": age,
        "Gender": encoders["Gender"].transform([gender])[0],
        "Tenure": tenure,
        "Usage Frequency": usage_freq,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Subscription Type": encoders["Subscription Type"].transform([subscription_type])[0],
        "Contract Length": encoders["Contract Length"].transform([contract_length])[0],
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
    }
    input_df = pd.DataFrame([row])
    # making sure columns are in the exact same order the model was trained on
    input_df = input_df[feature_columns]
    return input_df

if predict_btn:
    input_df = build_input_df()
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn (class 1)

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **CHURN** (probability: {probability:.1%})")
    else:
        st.success(f"✅ This customer is likely to **STAY** (churn probability: {probability:.1%})")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("See the input values used for this prediction"):
        st.dataframe(input_df)

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to get a result.")

st.markdown("---")
st.caption(
    "Model: Random Forest (tuned with GridSearchCV, ~99% F1 score on test data). "
    "Built for Epochs '26 - Assignment 9."
)
