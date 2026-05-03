import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")

st.title("🚀 Bitcoin Price Prediction Dashboard")
st.markdown("Upload a dataset and predict future Bitcoin prices using Machine Learning.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # ---------------- COLUMN SELECTION ----------------
    st.subheader("⚙️ Select Columns")

    col1, col2 = st.columns(2)

    with col1:
        date_col = st.selectbox("Select Date Column", data.columns)

    with col2:
        price_col = st.selectbox("Select Price Column", data.columns)

    # ---------------- DATA PROCESSING ----------------
    try:
        data[date_col] = pd.to_datetime(data[date_col])
    except:
        st.error("❌ Selected date column is not valid.")
        st.stop()

    data = data.sort_values(by=date_col)
    data["Days"] = (data[date_col] - data[date_col].min()).dt.days

    X = data[["Days"]]
    y = data[price_col]

    # ---------------- MODEL ----------------
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    # ---------------- METRICS ----------------
    r2 = model.score(X, y)
    mse = mean_squared_error(y, y_pred)

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(r2, 3))
    col2.metric("Mean Squared Error", f"{mse:,.2f}")

    # ---------------- PREDICTION INPUT ----------------
    st.subheader("🔮 Predict Future Price")

    future_days = st.slider("Days into the future", 1, 365, 30)

    future_input = pd.DataFrame({
        "Days": [data["Days"].max() + future_days]
    })

    prediction = model.predict(future_input)
    predicted_price = float(prediction[0])

    st.success(f"💰 Predicted Bitcoin price after {future_days} days: ${predicted_price:,.2f}")

    # ---------------- GRAPH ----------------
    st.subheader("📉 Visualization")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Scatter plot
    ax.scatter(X, y, color="blue", alpha=0.6, label="Historical Data")

    # Safe prediction range (fix overflow)
    max_day = int(data["Days"].max() + future_days)
    future_range = np.linspace(0, max_day, 500).reshape(-1, 1)
    future_pred = model.predict(future_range)

    # Line plot
    ax.plot(future_range, future_pred, color="red", linewidth=2, label="Prediction Trend")

    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_title("Bitcoin Price Prediction Trend")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to get started")
