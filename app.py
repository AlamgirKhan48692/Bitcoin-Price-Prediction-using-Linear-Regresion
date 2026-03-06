import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Bitcoin Price Prediction using Machine Learning")

st.write("Upload a Bitcoin historical dataset to predict future prices.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Bitcoin price dataset (CSV)", type="csv")

if uploaded_file is not None:

    # Read dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.write("Columns detected:", list(data.columns))

    # Detect date column automatically
    date_col = None
    for col in data.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_col = col
            break

    # Detect price column automatically
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

    # Sort dataset by date
    data = data.sort_values(by=date_col)

    # Create numeric timeline
    data["Days"] = (data[date_col] - data[date_col].min()).dt.days

    X = data[["Days"]]
    y = data[price_col]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Model accuracy
    score = model.score(X, y)

    st.subheader("Model Performance")
    st.write("Model R² Score:", round(score, 3))

    st.subheader("Prediction Settings")

    # Slider for future prediction
    future_days = st.slider("Days into the future", 1, 365, 30)

    future_input = pd.DataFrame({
        "Days": [data["Days"].max() + future_days]
    })

    prediction = model.predict(future_input)

    predicted_price = float(prediction[0])

    st.subheader("Predicted Bitcoin Price")

    st.success(f"Predicted price after {future_days} days: ${predicted_price:,.2f}")

    # Plot historical + prediction trend
    fig, ax = plt.subplots()

    ax.scatter(X, y, label="Historical Data")

    future_range = np.arange(0, data["Days"].max() + future_days).reshape(-1,1)
    future_pred = model.predict(future_range)

    ax.plot(future_range, future_pred, color="red", label="Prediction Trend")

    ax.set_xlabel("Days")
    ax.set_ylabel("Bitcoin Price")
    ax.set_title("Bitcoin Price Prediction Trend")

    ax.legend()

    st.pyplot(fig)
