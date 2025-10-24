import pandas as pd
import numpy as np
import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import StandardScaler
from datetime import datetime

# Load data
df = pd.read_csv('/Users/obanijesuadeyemo/Downloads/Fraud Detection App/Data/fraudTrain.csv')

# Define numerical and categorical features
CAT_FEATURES = ["merchant", "category", "gender", "job"]
NUM_FEATURES = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

# 1. ENCODING (Fit and Transform on Training Data)
# Store encoders/scalers in a dictionary for persistence in a real application.
# For this simple script, we fit them globally.
for feature in CAT_FEATURES:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])

# 2. SCALING (Fit and Transform on Training Data)
scaler = StandardScaler()
# Fit and transform only the numerical features
df[NUM_FEATURES] = scaler.fit_transform(df[NUM_FEATURES])

# Split features and target
x = df[ALL_FEATURES].values # Use the scaled and encoded features
y = df["is_fraud"].values

# Train model
# Training is now done on SCALED data!
model = SVC(probability=True)
model.fit(x, y)

# --- Streamlit App Starts Here ---
st.title("💳 Fraud Detection App")
st.write("This app predicts whether a transaction is fraudulent or legitimate based on key transaction details.")

st.sidebar.header("User Input Parameters")

# ... (user_input_features function remains the same as it collects raw values)
def user_input_features():
    # Non-predictive fields (for data entry realism)
    trans_date_trans_time = st.sidebar.date_input("Transaction Date & Time", datetime.now())
    cc_num = st.sidebar.text_input("Credit Card Number", "1234567890123456")
    first = st.sidebar.text_input("First Name")
    last = st.sidebar.text_input("Last Name")
    street = st.sidebar.text_input("Street")
    city = st.sidebar.text_input("City")
    state = st.sidebar.text_input("State")
    zip_code = st.sidebar.text_input("ZIP Code")
    dob = st.sidebar.date_input("Date of Birth")
    trans_num = st.sidebar.text_input("Transaction Number", "T123456789")

    # Predictive fields (used for model) - Assuming these are the encoded integer values
    # NOTE: The user input needs to be the actual encoded ID they would have gotten 
    # from the original merchant name, category name, etc. Using a slider on the 
    # encoded range is a simplification for a demo.
    merchant = st.sidebar.slider("Merchant (encoded ID)", int(df["merchant"].min()), int(df["merchant"].max()), step=1)
    category = st.sidebar.slider("Category (encoded ID)", int(df["category"].min()), int(df["category"].max()), step=1)
    gender = st.sidebar.slider("Gender (encoded ID: Male=1, Female=0)", int(df["gender"].min()), int(df["gender"].max()), step=1)
    job = st.sidebar.slider("Job (encoded ID)", int(df["job"].min()), int(df["job"].max()), step=1)
    amt = st.sidebar.slider("Transaction Amount", df['amt'].min(), df['amt'].max(), step=0.01)
    lat = st.sidebar.slider("Latitude", df['lat'].min(), df['lat'].max(), step=0.001)
    long = st.sidebar.slider("Longitude", df['long'].min(), df['long'].max(), step=0.001)
    city_pop = st.sidebar.slider("City Population", df['city_pop'].min(), df['city_pop'].max(), step=1)
    unix_time = st.sidebar.slider("Unix Time", df['unix_time'].min(), df['unix_time'].max(), step=1000)
    merch_lat = st.sidebar.slider("Merchant Latitude", df['merch_lat'].min(), df['merch_lat'].max(), step=0.001)
    merch_long = st.sidebar.slider("Merchant Longitude", df['merch_long'].min(), df['merch_long'].max(), step=0.001)
    
    data = {
        # ... (all other features)
        "merchant": merchant, "category": category, "amt": amt, "gender": gender, 
        "lat": lat, "long": long, "city_pop": city_pop, "job": job, 
        "unix_time": unix_time, "merch_lat": merch_lat, "merch_long": merch_long
    }
    # ... (collecting all inputs into features DataFrame)
    
    features = pd.DataFrame(data, index=[0])
    return features


# Get user input
input_df = user_input_features()

# Extract only columns used by the model
model_features = input_df[ALL_FEATURES].copy() # Use .copy() for safe manipulation

# 3. APPLY SCALING (Crucial step for prediction)
# Transform only the numerical features of the user input using the fitted scaler
model_features[NUM_FEATURES] = scaler.transform(model_features[NUM_FEATURES])

# Show user inputs (showing the pre-scaled input for clarity)
st.subheader("🔍 Entered Transaction Details (Raw Values)")
st.write(input_df)

# Predict fraud
prediction = model.predict(model_features)
prediction_proba = model.predict_proba(model_features)[0]

# ... (Display results)
st.subheader("🧠 Prediction Result")
if prediction[0] == 1:
    st.error("🚨 This transaction is predicted to be **FRAUDULENT!**")
else:
    st.success("✅ This transaction is predicted to be **LEGITIMATE.**")

st.subheader("📊 Prediction Confidence")
st.write(f"Legit: {prediction_proba[0]:.4f} | Fraud: {prediction_proba[1]:.4f}")