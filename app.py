import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Bitcoin Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload Bitcoin price dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Show columns so user knows dataset structure
    st.write("Columns in dataset:", list(data.columns))

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
        st.error("Could not detect Date or Price column automatically. Please check your dataset.")
    else:
        # Convert date column
        data[date_col] = pd.to_datetime(data[date_col])

        # Convert to numeric timeline
        data["Days"] = (data[date_col] - data[date_col].min()).dt.days

        X = data[["Days"]]
        y = data[price_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        future_days = st.slider("Days in future for prediction", 1, 365, 30)

        future_value = model.predict([[data["Days"].max() + future_days]])

        st.subheader("Predicted Bitcoin Price")
        st.write(float(future_value[0]))

        # Plot results
        fig, ax = plt.subplots()
        ax.scatter(X, y)
        ax.plot(X, model.predict(X), color="red")
        ax.set_xlabel("Days")
        ax.set_ylabel("Bitcoin Price")

        st.pyplot(fig)