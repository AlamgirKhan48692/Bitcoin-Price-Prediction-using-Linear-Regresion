import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bitcoin Predictor", layout="wide")

st.title("🚀 Bitcoin Price Prediction Dashboard")

uploaded_file = st.file_uploader("📂 Upload CSV file", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("Columns detected:", list(data.columns))

    # -------- AUTO DETECT --------
    date_guess = None
    price_guess = None

    for col in data.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_guess = col
        if "close" in col.lower() or "price" in col.lower():
            price_guess = col

    # -------- SELECT COLUMNS --------
    st.subheader("⚙️ Select Columns")

    col1, col2 = st.columns(2)

    with col1:
        date_col = st.selectbox("Select Date Column", data.columns,
                                index=data.columns.get_loc(date_guess) if date_guess else 0)

    with col2:
        price_col = st.selectbox("Select Price Column", data.columns,
                                 index=data.columns.get_loc(price_guess) if price_guess else 1)

    # -------- VALIDATION --------
    if date_col == price_col:
        st.error("❌ Date and Price columns must be different")
        st.stop()

    # -------- CLEAN DATA --------
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data[price_col] = pd.to_numeric(data[price_col], errors='coerce')

    data = data.dropna(subset=[date_col, price_col])

    if len(data) < 10:
        st.error("❌ Not enough valid data")
        st.stop()

    data = data.sort_values(by=date_col)

    # -------- CREATE DAYS --------
    data["Days"] = (data[date_col] - data[date_col].min()).dt.days.astype(float)

    X = data["Days"].values.reshape(-1, 1)
    y = data[price_col].values

    # -------- MODEL --------
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    # -------- METRICS --------
    r2 = model.score(X, y)
    mse = mean_squared_error(y, y_pred)

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(r2, 3))
    col2.metric("MSE", f"{mse:,.2f}")

    # -------- FUTURE PREDICTION --------
    st.subheader("🔮 Predict Future Price")

    future_days = st.slider("Days into the future", 1, 365, 30)

    future_value = np.array([[X.max() + future_days]])
    predicted_price = model.predict(future_value)[0]

    st.success(f"💰 Predicted price after {future_days} days: ${predicted_price:,.2f}")

    # -------- GRAPH --------
    st.subheader("📉 Visualization")

    fig, ax = plt.subplots(figsize=(10,5))

    # Scatter
    ax.scatter(X.flatten(), y, alpha=0.6, label="Historical Data")

    # Smooth line
    x_line = np.linspace(X.min(), X.max() + future_days, 300).reshape(-1,1)
    y_line = model.predict(x_line)

    ax.plot(x_line, y_line, linewidth=2, label="Prediction Trend")

    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_title("Bitcoin Price Prediction Trend")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to get started")
