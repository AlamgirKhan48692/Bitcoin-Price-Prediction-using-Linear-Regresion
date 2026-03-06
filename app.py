import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Bitcoin Price Prediction App")

st.write("Upload a Bitcoin historical dataset to predict future prices.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Bitcoin price dataset (CSV)", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.write("Columns detected:", list(data.columns))

    # Detect date column
    date_col = None
    for col in data.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_col = col
            break

    # Detect price column
    price_col = None
    for col in data.columns:
        if "close" in col.lower() or "price" in col.lower():
            price_col = col
            break

    if date_col is None or price_col is None:
        st.error("Date or Price column could not be detected automatically.")
        st.stop()

    # Convert date column
    data[date_col] = pd.to_datetime(data[date_col])

    # Create numeric timeline
    data["Days"] = (data[date_col] - data[date_col].min()).dt.days

    X = data[["Days"]]
    y = data[price_col]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Prediction Settings")

    future_days = st.slider("Days into the future", 1, 365, 30)

    future_input = pd.DataFrame({
        "Days":[data["Days"].max() + future_days]
    })

    prediction = model.predict(future_input)

    predicted_price = float(prediction[0])

    st.subheader("Predicted Bitcoin Price")

    st.success(f"Predicted price after {future_days} days: ${predicted_price:,.2f}")

    # Plot
    fig, ax = plt.subplots()

    ax.scatter(X, y, label="Historical Data")
    ax.plot(X, model.predict(X), color="red", label="Regression Line")

    ax.set_xlabel("Days")
    ax.set_ylabel("Bitcoin Price")
    ax.set_title("Bitcoin Price Trend")

    ax.legend()

    st.pyplot(fig)