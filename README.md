# Customer Churn Predictor - Web App

**Name:** EB Fathima Suhana
**MUID:** fathimasuhana@mulearn

🌐 **Deployment Link:** [streamlit app](https://customer-churn-predictor---web-app-6mtlmsv2t5xamgotgxqfgd.streamlit.app/)

## Project Overview

This was for Assignment 9 of Epochs '26 - the task was to take a model from an earlier assignment and actually turn it into a working web app that people can use, instead of it just living inside a notebook.

I reused the **Random Forest churn prediction model** from Assignment 8 (Customer Churn Prediction). The app lets someone punch in a customer's details (age, tenure, support calls, payment delay, etc.) through a simple sidebar form, and it instantly tells you whether the model thinks that customer is going to churn or not, along with the churn probability.

## What's in this repo

```
├── app.py                  # the actual streamlit app
├── train_model.py          # script that retrains the model + saves it to model/
├── model/
│   ├── churn_model.pkl     # trained Random Forest model
│   ├── scaler.pkl          # StandardScaler fit on training data
│   ├── encoders.pkl        # LabelEncoders for Gender / Subscription Type / Contract Length
│   └── feature_columns.pkl # exact column order the model expects
├── requirements.txt
└── README.md
```

## Deployment Approach

Went with **Streamlit Community Cloud** since its free and honestly the easiest option to just push a GitHub repo and get a live link, no server config or docker stuff needed like Render.

Steps I followed:
1. Trained the Random Forest model (reusing the same code/approach from Assignment 8 - sampled 20k rows from the churn dataset, same train/test split, same best hyperparameters GridSearchCV found: `n_estimators=200`, `max_depth=None`, `min_samples_split=2`).
2. Instead of retraining the model every time the app runs (which would be slow and pointless), I saved the trained model + scaler + encoders using `joblib` so the app just loads them once when it starts.
3. Built the UI with Streamlit - sliders and dropdowns in the sidebar for each input feature, and a "Predict Churn" button that runs the saved model on whatever the user entered and shows the result (churn / stay + probability).
4. Pushed everything (`app.py`, `train_model.py`, `model/` folder, `requirements.txt`) to a public GitHub repo.
5. Connected the repo to Streamlit Community Cloud and deployed it from there.

## Key Observations

- The model itself performs really well (~99% F1 score on the test set from Assignment 8), so predictions in the app feel pretty confident/decisive most of the time - like it doesn't sit on the fence around 50% very often, its usually leaning strongly one way or the other.
- Support Calls and Payment Delay have a noticeably big effect on the prediction - cranking those sliders up in the app flips the prediction to "churn" pretty quickly compared to changing other fields, which lines up with what the feature importance chart showed in Assignment 8.
- Caching the model loading with `@st.cache_resource` made a big difference in how snappy the app feels - without it, streamlit was reloading the pickle files on every single interaction which felt kinda slow.

## Challenges Faced

- Initially I was going to just load the raw dataset and train the model live inside the app, but that was way too slow to run every time someone opens the app or even just moves a slider. Switched to pre-training the model separately (`train_model.py`) and just loading the saved `.pkl` files in `app.py`, which fixed it.
- Had to make sure the input dataframe columns are in the **exact same order** as what the model was trained on, otherwise the predictions come out completely wrong even though there's no error thrown. Fixed this by saving `feature_columns.pkl` during training and reusing it in the app.
- Encoding the dropdown values (like Gender, Subscription Type) properly was a bit fiddly - had to save the actual `LabelEncoder` objects from training and reuse them in the app instead of hardcoding the encoding mapping, since hardcoding is risky if I ever retrain and the encoding changes.

## Future Improvements

- Add a way to upload a whole CSV of customers and get bulk predictions instead of one at a time.
- Show a simple explanation of *why* the model predicted churn for that specific customer (like using SHAP values) instead of just the probability number.
- Add some input validation / warnings if someone enters weird combos (like 0 tenure but really high total spend).
- Try deploying the same model on Hugging Face Spaces too just to compare how it feels vs Streamlit Cloud.
- Retrain on the full 440k row dataset instead of the 20k sample if I get access to more compute, might make the model even better.
