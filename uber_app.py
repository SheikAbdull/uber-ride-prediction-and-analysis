import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("uber_fare_model.pkl")

st.title("🚖 Uber Ride Fare Prediction and Analysis")

# Sidebar input
st.sidebar.header("Enter Ride Details")
pickup_latitude = st.sidebar.slider("Pickup Latitude", 40.70, 40.85, 40.75)
pickup_longitude = st.sidebar.slider("Pickup Longitude", -74.05, -73.90, -73.95)
dropoff_latitude = st.sidebar.slider("Dropoff Latitude", 40.70, 40.85, 40.75)
dropoff_longitude = st.sidebar.slider("Dropoff Longitude", -74.05, -73.90, -73.95)
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)
hour = st.sidebar.slider("Pickup Hour", 0, 23, 12)

# Predict fare
if st.sidebar.button("Predict Fare"):
    features = np.array([[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, hour]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Fare: ${prediction:.2f}")

# Load and analyze data
st.header("📊 Historical Ride Data Analysis")
df = pd.read_csv("uber_data.csv")
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour

# Show graphs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Fare vs Passenger Count")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='passenger_count', y='fare_amount', data=df, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Fare Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['fare_amount'], bins=30, ax=ax2, kde=True)
    st.pyplot(fig2)

