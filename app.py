import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Bitcoin Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload Bitcoin price dataset (CSV)")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Example: using 'Date' and 'Close'
    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days

    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_days = st.slider("Days in future for prediction", 1, 365, 30)

    future_value = model.predict([[data['Days'].max() + future_days]])

    st.subheader("Predicted Bitcoin Price")
    st.write(future_value[0])

    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, model.predict(X), color='red')
    ax.set_xlabel("Days")
    ax.set_ylabel("Bitcoin Price")

    st.pyplot(fig)